# OAI gNB 独立升级环境：B210 + RM520N-GL

这部分连接真实 USRP、5GC 和 UE。MATLAB 仿真仍在仓库原有的 `oran-sim/` 中。

基站固定在官方 **2026.w42**；源码提交与容器摘要见
[upstream.lock.json](upstream.lock.json)。容器内是独立的 UHD 和运行库，
宿主机原来的 OAI、UHD、5GC、订阅数据库与 SIM 保留。更换 gNB 不需要重新写 SIM。

2026-10-20 已在实验主机完成验证：旧版为 `2025.w38`，新版 `2026.w42` 在相同
5GC 和 SIM 下完成 40 MHz 接入与约 35 秒下载，校验了 8,784 帧、无校验错误，
去掉预热后的接收均值约 33.9 Mbps。这是连通性验证，不是峰值吞吐基准。
手动回退旧版、模块绑定下载、再次切换新版均已实测；最后保留新版运行。
详见[脱敏验证记录](validation/2026-10-20.json)。
后续已完成[10–50 MHz 逐档测试及英文结果图](validation/bandwidth-2026-10-20/README.md)，包括无法运行的档位与恢复记录。此记录是当日快照，实时状态用下文的
`status` 命令查看。

## 环境与版本

在 Linux USRP 主机运行以下命令。需要 Docker Compose、Python 3、Git，以及已在运行的旧版
`nr-softmodem` systemd 服务。新旧 gNB 不能同时占用同一块 B210。

状态目录默认为 `~/oai-gnb-next/`，包含固定提交的上游源码、UHD 固件缓存、当前配置、
旧基站启动快照和本地日志。它位于 Git 仓库之外。

```text
Open-RAN/lab/oai-gnb/       Git 管理的运行脚本、配置模板、版本锁和教程
~/oai-gnb-next/src/         固定提交的完整 OAI 源码
~/oai-gnb-next/gnb.conf     本机实际使用的配置
~/oai-gnb-next/uhd-images/  容器 UHD 对应的 B210 固件缓存
~/oai-gnb-next/legacy.json  旧基站的精确启动参数与工作目录
~/oai-gnb-next/logs/        本地运行日志，不上传 GitHub
```

不要在这个公开仓库中加入 SSH 密码、SIM 的 K/OPc、订阅数据库或未脱敏的 UE 日志。
本项目没有重新部署核心网，也没有生成新的订阅身份或密钥。

## 安装与准备

```bash
git clone --branch codex/oai-gnb-upgrade https://github.com/Kilo377/Open-RAN.git ~/Open-RAN
cd ~/Open-RAN
python3 lab/oai-gnb/scripts/labctl.py prepare \
  --legacy-unit oai-pwr-20261020-c1-baseline.service
```

`--legacy-unit` 是这台实验主机当前的旧基站服务名。其他主机应替换为其实际服务名。
准备阶段下载并校验源码提交、拉取固定摘要镜像、缓存 B210 固件，并检查 Compose 与二进制选项。
它不会打开 B210，旧基站继续运行。

`configs/gnb.sa.band78.40MHz.b210.conf` 来源于锁定提交的官方模板。
当前初始配置为 n78、3.6192 GHz、30 kHz SCS、106 PRB（40 MHz）、1 TX / 1 RX、
`att_tx=12`。它保留实验室现有的 PLMN `001/01`、TAC `1`、SST `1`、AMF
`192.168.70.132` 和主机侧 N2/N3 地址 `192.168.70.129/24`。
这些参数需要与实际核心网匹配；需要改动时编辑状态目录中的 `gnb.conf`。

使用新版模板保留了新的参数形式，例如 `do_SRS="periodic"`；不要直接覆盖回旧版模板。

## 切换、验证和保留新版

```bash
cd ~/Open-RAN
sudo python3 lab/oai-gnb/scripts/labctl.py activate
python3 lab/oai-gnb/scripts/labctl.py status
```

切换前会先创建独立的 systemd 回退定时器。旧 gNB 停止后才启动新容器。
脚本检查 NG Setup、RF ready、40 MHz、106 PRB 和正常 UHD TX gain。
启动失败会执行回退；未接受的试运行会在约 10 分钟后回退。

本实验室已有 OAI Link Monitor：host 上的 HTTP 服务器向 AGX 发资源，AGX 的数据 socket
绑定 RM520N-GL USB 网卡，Wi-Fi 只用于管理。可以直接用它做 35 秒验证：

```bash
python3 lab/oai-gnb/scripts/labctl.py smoke-test \
  --monitor-url http://127.0.0.1:18765 --duration 35
sudo python3 lab/oai-gnb/scripts/labctl.py accept
```

验证必须有真实的已校验负载、正确的模块网卡绑定，以及持续接收记录。
`accept` 只接受当前容器和当前配置对应的通过记录，然后取消回退定时器、保持新版运行。
这项测试验证接入与数据路径，不等同于峰值吞吐测试。

没有安装这个实验室监测器时，可先按[官方商用 UE 教程](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/2026.w42/doc/NR_SA_Tutorial_COTS_UE.md)
完成 UE 与流量测试。当前 `smoke-test`/`accept` 的自动验证接口依赖该监测器；
不要将缺少测试依赖误认为新 gNB 或 SIM 不兼容。

新容器没有配置开机自动启动。日常开机后的启动入口见 [STARTUP.md](STARTUP.md)，
包含核心网、新版 gNB、AGX 接收服务、UI 和接入失败时的检查方法。

## 回到旧版

```bash
cd ~/Open-RAN
sudo python3 lab/oai-gnb/scripts/labctl.py restore
```

该命令只停止本项目的 `oai-gnb-next` 容器，然后恢复保存的旧基站服务。
如果旧服务是停止后消失的 transient unit，会用快照中的参数重新创建它。
核心网容器、MySQL 数据及 SIM 不参与切换。旧版本重新接入后可继续用原来的 UI 测试。

## 同步与继续开发

在 Mac 或其他开发机编辑、提交并推送这个分支，然后在 USRP 主机同步：

```bash
cd ~/Open-RAN
git pull --ff-only origin codex/oai-gnb-upgrade
```

运行中的镜像按摘要固定，`git pull` 不会自动切换基站。
修改上游源码也不会自动改变预编译镜像。源码开发时应单独构建新镜像，保存补丁、
更新版本锁和测试记录，再用相同的回退流程验证。
上游构建入口见[固定版本的 BUILD.md](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/2026.w42/doc/BUILD.md)
以及源码中的 `docker/Dockerfile.base.ubuntu`、`Dockerfile.build.ubuntu` 和 `Dockerfile.gNB.ubuntu`。
当前安装使用官方预编译镜像，完整源码重编译不属于这次已验证内容。

升级目标与实测支持要区分：源码检查发现 `2026.w42` 的 30 kHz SCS 采样率映射仍未列出
38/65/78 PRB（15/25/30 MHz），B210 专用驱动分支仍未列出 61.44 MS/s。
因此升级本身不会自动补齐这些带宽。后续可在本分支维护适配补丁，再逐档实测，
不要把理论支持、能启动和通过完整 UE 下载测试混为一谈。

脱敏的本机验证结果保存于 [validation/](validation/)；原始日志只保留在状态目录。

## 来源

- [OAI 2026.w42 固定源码](https://gitlab.eurecom.fr/oai/openairinterface5g/-/tree/2026.w42)
- [官方 gNB 镜像](https://hub.docker.com/r/oaisoftwarealliance/oai-gnb/tags?name=2026.w42)
- [OAI 许可](https://gitlab.eurecom.fr/oai/openairinterface5g/-/blob/2026.w42/LICENSE)

配置模板保留了上游 `SPDX-License-Identifier: LicenseRef-CSSL-1.0` 标识。
