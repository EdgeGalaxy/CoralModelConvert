"""RKNN conversion service extracted and adapted from the original implementation"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from loguru import logger

from ..exceptions import (
    ModelLoadError, 
    ModelBuildError, 
    ModelExportError, 
    ModelConversionError,
    InvalidParameterError
)
from ..config import RKNN_SUPPORTED_PLATFORMS, RKNN_SUPPORTED_QUANT_ALGOS


class RKNNConverter:
    """Convert ONNX models to RKNN format with various quantization options"""
    
    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        output_dir: Union[str, Path],
        dataset_file: Optional[str] = None,
        target_platform: str = "rk3588",
        hybrid_quant: bool = True,
        quantized_algorithm: str = "normal",
        quantized_method: str = "channel",
        optimization_level: int = 3,
        rknn_batchsize: Optional[int] = None,
        with_acc_analysis: bool = False,
        step: str = "onestep"
    ):
        self._validate_parameters(target_platform, step, quantized_algorithm)
        
        try:
            from rknn.api import RKNN
        except ImportError:
            raise ModelConversionError("RKNN library not available. Please install rknn-api.")

        self.onnx_model = Path(onnx_model_path)
        self.output_dir = Path(output_dir)
        self.dataset = dataset_file
        self.target_platform = target_platform
        self.hybrid_quant = hybrid_quant
        self.quantized_algorithm = quantized_algorithm
        self.optimization_level = optimization_level
        self.do_quant = bool(dataset_file)
        self.rknn_batchsize = rknn_batchsize
        self.with_acc_analysis = with_acc_analysis
        self.step = step

        self.mean_values = [[0, 0, 0]]
        self.std_values = [[255, 255, 255]]
        self.rknn = RKNN(verbose=False)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_parameters(self, target_platform: str, step: str, quantized_algorithm: str):
        """Validate input parameters"""
        if target_platform not in RKNN_SUPPORTED_PLATFORMS:
            raise InvalidParameterError(
                f"Unsupported platform: {target_platform}. Must be one of {RKNN_SUPPORTED_PLATFORMS}"
            )
        
        supported_steps = ["onestep", "step1", "step2"]
        if step not in supported_steps:
            raise InvalidParameterError(
                f"Unsupported step: {step}. Must be one of {supported_steps}"
            )
        
        if quantized_algorithm not in RKNN_SUPPORTED_QUANT_ALGOS:
            raise InvalidParameterError(
                f"Unsupported quantization algorithm: {quantized_algorithm}. Must be one of {RKNN_SUPPORTED_QUANT_ALGOS}"
            )

    def convert(self) -> Dict[str, Any]:
        """Main conversion method that handles different conversion strategies"""
        try:
            logger.info(f"Starting RKNN conversion for {self.onnx_model.name}")
            
            if self.hybrid_quant and self.step == "onestep":
                logger.info("Starting hybrid quantization...")
                self._hybrid_quantization_step1()
                result = self._hybrid_quantization_step2()
            elif self.step == "onestep":
                logger.info(f"Starting normal quantization, {'using int8 quantization' if self.do_quant else 'using fp16 quantization'}")
                result = self._model_quantization()
            elif self.step == "step1":
                result = self._hybrid_quantization_step1(with_acc_analysis=self.with_acc_analysis)
            elif self.step == "step2":
                result = self._hybrid_quantization_step2()
            else:
                raise ModelConversionError("Only when `hybrid_quant` is True can `step` be set to `step1` and `step2`")
            
            logger.info("RKNN conversion completed successfully")
            return result
            
        except Exception as e:
            logger.exception("Model conversion failed")
            raise e
        finally:
            self.rknn.release()

    def _model_quantization(self) -> Dict[str, Any]:
        """Internal method to handle model quantization"""
        logger.debug("Configuring model...")
        self.rknn.config(
            mean_values=self.mean_values,
            std_values=self.std_values,
            target_platform=self.target_platform,
            optimization_level=self.optimization_level,
            quantized_algorithm=self.quantized_algorithm
        )

        logger.debug("Loading model...")
        if self.rknn.load_onnx(model=str(self.onnx_model)) != 0:
            raise ModelLoadError("Failed to load ONNX model")

        logger.debug("Building model...")
        if self.rknn.build(do_quantization=self.do_quant, dataset=self.dataset, rknn_batch_size=self.rknn_batchsize) != 0:
            raise ModelBuildError("Failed to build model")

        output_path = self.output_dir / f"{self.onnx_model.stem}.rknn"
        logger.debug(f"Exporting RKNN model to {output_path}...")
        if self.rknn.export_rknn(str(output_path)) != 0:
            raise ModelExportError("Failed to export RKNN model")
        
        logger.info(f"Successfully exported RKNN model to {output_path}")
        
        return {
            "output_path": str(output_path),
            "quantization_type": "int8" if self.do_quant else "fp16",
            "platform": self.target_platform,
            "algorithm": self.quantized_algorithm
        }

    def _hybrid_quantization_step1(self, with_acc_analysis: bool = False) -> Dict[str, Any]:
        """First step of hybrid quantization"""
        logger.debug("Configuring model for hybrid quantization step 1...")
        self.rknn.config(
            mean_values=self.mean_values,
            std_values=self.std_values,
            target_platform=self.target_platform,
            optimization_level=self.optimization_level,
            quantized_algorithm=self.quantized_algorithm
        )

        logger.debug("Loading model...")
        if self.rknn.load_onnx(model=str(self.onnx_model)) != 0:
            raise ModelLoadError("Failed to load ONNX model")

        logger.debug("Running hybrid quantization step 1...")
        if self.rknn.hybrid_quantization_step1(dataset=self.dataset, proposal=True, rknn_batch_size=self.rknn_batchsize) != 0:
            raise ModelConversionError("Hybrid quantization step 1 failed")

        # Move generated files to output directory
        generated_files = []
        for ext in [".data", ".model", ".quantization.cfg"]:
            src = self.onnx_model.stem + ext
            dst = self.output_dir / src
            if Path(src).exists():
                shutil.move(src, dst)
                generated_files.append(str(dst))
                logger.debug(f"Moved {src} to {dst}")

        if with_acc_analysis:
            self.rknn.release()
            from rknn.api import RKNN
            self.rknn = RKNN(verbose=False)
            self._accuracy_analysis()

        return {
            "step": "step1_completed",
            "generated_files": generated_files,
            "platform": self.target_platform,
            "accuracy_analysis": with_acc_analysis
        }

    def _hybrid_quantization_step2(self) -> Dict[str, Any]:
        """Second step of hybrid quantization"""
        model_input = self.output_dir / f"{self.onnx_model.stem}.model"
        data_input = self.output_dir / f"{self.onnx_model.stem}.data"
        model_quantization_cfg = self.output_dir / f"{self.onnx_model.stem}.quantization.cfg"

        required_files = [
            (model_input, "model file"),
            (data_input, "data file"),
            (model_quantization_cfg, "quantization config file")
        ]

        for file_path, file_type in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required {file_type} not found at {file_path}. Please run hybrid_quantization_step1 first."
                )

        logger.debug("Running hybrid quantization step 2...")
        if self.rknn.hybrid_quantization_step2(
            model_input=str(model_input),
            data_input=str(data_input),
            model_quantization_cfg=str(model_quantization_cfg)
        ) != 0:
            raise ModelConversionError("Hybrid quantization step 2 failed")

        output_path = self.output_dir / f"{self.onnx_model.stem}.rknn"
        logger.debug(f"Exporting RKNN model to {output_path}...")
        if self.rknn.export_rknn(str(output_path)) != 0:
            raise ModelExportError("Failed to export RKNN model")
        
        logger.info(f"Successfully exported RKNN model to {output_path}")
        
        return {
            "output_path": str(output_path),
            "quantization_type": "hybrid_int8",
            "platform": self.target_platform,
            "algorithm": self.quantized_algorithm
        }

    def _accuracy_analysis(self) -> None:
        """Perform accuracy analysis on the model"""
        if not self.dataset:
            raise ValueError("Dataset file is required for accuracy analysis")

        self._model_quantization()

        with open(self.dataset) as f:
            img_path = f.readline().strip()
            if not img_path:
                raise ValueError("Unable to obtain a valid image from dataset file")

        logger.debug("Running accuracy analysis...")
        output_dir = self.output_dir / "snapshot"
        output_dir.mkdir(exist_ok=True)
        
        if self.rknn.accuracy_analysis(inputs=[img_path], output_dir=str(output_dir)) != 0:
            raise ModelConversionError("Accuracy analysis failed")

        # Clean up temporary RKNN model
        temp_rknn = self.output_dir / f"{self.onnx_model.stem}.rknn"
        if temp_rknn.exists():
            temp_rknn.unlink()