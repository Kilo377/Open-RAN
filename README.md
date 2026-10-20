# Open-RAN：OAI 实机实验

这个分支用于 USRP B210、OAI gNB 和 Quectel RM520N-GL / Jetson AGX 的真实 5G 接入与吞吐实验。
运行配置、启动脚本、教程和脱敏实测结果统一保存在 [`lab/oai-gnb/`](lab/oai-gnb/README.md)。

当前固定使用 OAI **2026.w42**，通过独立容器运行新版 gNB，复用实验室已有的 5GC 和 SIM。
已验证的正常配置为 n78、40 MHz / 106 PRB、30 kHz SCS、1 TX / 1 RX。
AGX 的测试下载 socket 绑定 RM520N-GL USB 网卡，Wi-Fi 用于管理报告。

首次准备环境请按[安装与版本说明](lab/oai-gnb/README.md)操作。
已经准备好环境、且 AGX 接收代理就绪时，在 USRP 主机执行：

```bash
cd ~/Open-RAN
git pull --ff-only origin codex/oai-gnb-upgrade
sudo python3 lab/oai-gnb/scripts/start_up.py
```

启动入口检查核心网健康状态，启动新版 gNB 和监测服务，并完成约 35 秒的真实下载验证。
主机 UI 地址为 `http://127.0.0.1:18765/`。
AGX 端启动、打开 UI、重新选网和故障检查见 [Start-up 教程](lab/oai-gnb/STARTUP.md)。

```text
lab/oai-gnb/
├── README.md          安装、切换、验证和回退
├── STARTUP.md         日常启动与排查
├── upstream.lock.json 固定源码版本和镜像摘要
├── compose.yaml       新版 gNB 容器
├── configs/           B210 配置模板
├── scripts/           启动与生命周期管理
└── validation/        脱敏验证记录、CSV 和结果图
```

[10–50 MHz 带宽实测](lab/oai-gnb/validation/bandwidth-2026-10-20/README.md)记录了通过与失败的全部档位。
无法运行的档位标为 N/A，不代表零吞吐；当前版本尚未适配所有标准带宽。

本仓库不保存 SSH 密码、SIM 密钥、订阅数据库或原始 UE 日志。
上游 OAI 配置保留原有许可证标识，版本与许可来源见[实验说明](lab/oai-gnb/README.md#来源)。
