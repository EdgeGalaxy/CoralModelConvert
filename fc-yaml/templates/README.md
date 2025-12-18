# CoralModelConvert - Aliyun FC 3.0 YAML 模板

本目录下存放的是 CoralModelConvert 在阿里云函数计算 3.0 上的函数 YAML「模板版本」，通过环境变量渲染后生成真正要导入/部署的 YAML。

## 目录说明

- `s-onnx2rknn.template.yaml`：onnx2rknn 函数模板（基于当前 `CoralModelConvert/s.yaml` 抽取）

渲染后对应生成：

- `CoralModelConvert/fc-yaml/s-onnx2rknn.yaml`

## 使用方式（envsubst 渲染）

1. 使用 `.env` 文件管理变量（推荐）：

   ```bash
   cd CoralModelConvert/fc-yaml/templates

   # 拷贝示例文件并按需修改
   cp .env.example .env

   # 将 .env 中的变量导入当前 shell
   set -a
   source .env
   set +a
   ```

   也可以跳过 `.env`，直接在当前 shell 中手动 `export` 各变量。

2. 使用 `envsubst` 渲染模板生成实际 YAML：

   ```bash
   cd CoralModelConvert/fc-yaml/templates

   envsubst < s-onnx2rknn.template.yaml > ../s-onnx2rknn.yaml
   ```

3. 将生成的 `CoralModelConvert/fc-yaml/s-onnx2rknn.yaml` 导入阿里云函数计算控制台，或按你的工作流进行部署。

## 变量说明（与模板字段对应）

- 项目 / 区域：
  - `FC_PROJECT_NAME`：YAML 中的 `name`
  - `OSS_REGION`：`props.region`

- 镜像与 OSS：
  - `IMAGE`：`customContainerConfig.image`
  - `OSS_ENDPOINT`：`environmentVariables.OSS_ENDPOINT`
  - `OSS_BUCKET`：`environmentVariables.OSS_BUCKET_NAME`

- 账号 / 资源组 / 日志：
  - `SERVICE_ROLE`：`props.role`
  - `FC_ACCOUNT_ID`：阿里云账号 ID（当前模板未直接使用，方便统一配置）
  - `RESOURCE_GROUP_ID`：`props.resourceGroupId`
  - `LOG_PROJECT` / `LOG_STORE`：`logConfig.project` / `logConfig.logstore`

- VPC：
  - `VPC_ID` / `VSWITCH_ID` / `SECURITY_GROUP_ID`：`vpcConfig` 中的各字段

- OSS 凭证（不要提交到仓库）：
  - `OSS_ACCESS_KEY_ID`
  - `OSS_ACCESS_KEY_SECRET`

## 手动修改的最小化建议

- 日常只需要修改：
  - 镜像版本：更新 `.env` 中的 `IMAGE`
  - 切换账号 / 区域 / VPC / OSS：更新 `.env` 中对应变量
- 配置调整（如 CPU/内存/磁盘/并发）仍然在模板 YAML 内修改即可，一般不需要频繁变更。

