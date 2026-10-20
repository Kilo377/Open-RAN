# Start up：开机后启动实验环境

适用于已完成本目录 README 中 `prepare`、并已安装 Link Monitor 的这套 B210 实验环境。
首次安装仍按 [README](README.md) 完成准备；本流程复用现有 5GC、SIM 与固定版本镜像。
正常配置为 OAI `2026.w42`、n78、40 MHz / 106 PRB、30 kHz SCS、`att_tx=12`。

2026-10-20 已实机验证停止 gNB 后启动、以及重复执行启动入口，均通过模块绑定下载验证。
重复启动保留了 gNB 的进程启动时间，核心网容器也未重启；本次没有重启主机操作系统。
详见[启动验证记录](validation/startup-2026-10-20.json)。

先连接 USRP 主机和 AGX，两台机器各保留一个终端。密码通过 SSH 提示输入。

```bash
# USRP host
ssh xiaolanliu@100.115.7.51

# 另一个终端：AGX / UE
ssh lboroldn@100.116.105.126
```

在 **AGX** 上启动接收代理。RM520N-GL 需通电，USB 网卡应有地址。

```bash
sudo systemctl start oai-link-agent.service
systemctl is-active oai-link-agent.service
ip -br addr show enx46b5bc4bcef0
```

在 **USRP 主机** 上同步并启动：

```bash
cd ~/Open-RAN
git pull --ff-only origin codex/oai-gnb-upgrade
sudo python3 lab/oai-gnb/scripts/start_up.py
```

可选的 `python3 lab/oai-gnb/scripts/start_up.py --check` 是只读依赖检查，需要 Docker daemon 已运行。启动入口会启动 Docker，
复用 `~/oai-cn5g/` 的本地核心网镜像与现有容器，等待核心网健康，再启动主机监测服务和新版 gNB。
旧版 gNB 仍运行时，沿用 `labctl activate` 的独立 10 分钟回退保护；已经运行的新版容器不会被重启。
AGX 代理就绪后执行约 35 秒的真实下载验证，只有持续接收、模块网卡绑定和负载校验通过，
才接受新版运行并取消回退定时器。启动前请结束 UI 中正在运行的测试。

AGX 缺失、UE 未注册、核心网不健康等情况会输出具体错误。
新 gNB 刚被激活而验证未通过时，回退定时器保留；处理问题后及时重跑启动入口，
或按 README 中的 `smoke-test` / `accept` 完成验证。不要把启动失败提示当成已经接通。

这套入口是手动启动，不会安装新的开机自启服务。现有两个 Link Monitor 服务保留原有的自启设置。

在 **USRP 主机的桌面终端** 打开 UI：

```bash
firefox http://127.0.0.1:18765/
```

也可双击主机桌面的 **5G Link Monitor** 快捷方式。
从同一 Tailscale 网络中的其他设备访问 `http://100.115.7.51:18765/`。
页面应显示 gNB running、AGX online 和 RM520N-GL ready。点击 Start 开始连续下载，
出现 Module-bound data verified 后再进行遮挡实验。Wi-Fi 仅承载管理报告，下载 socket 绑定模块网卡。

如果模块没有回到实验小区，在**已经可用的模块 AT 终端**检查并重新选网：

```text
AT+COPS?
AT+C5GREG?
AT+QENG="servingcell"
AT+COPS=1,2,"00101"
AT+C5GREG?
```

`00101` 对应本实验的 PLMN；其他实验网络应使用自己的 PLMN。
上述重新选网命令会选择手动模式；先记录原有选网模式，需要恢复自动模式时使用 `AT+COPS=0`。
5G 注册状态应为 `1`（本地）或网络允许时的 `5`（漫游），并继续用下载验证实际数据链路。
本次实测在不兼容带宽试验后曾进入其他 PLMN 的受限服务状态，重新选择实验网络后恢复。
AGX 当前未生成 `/dev/ttyUSB*`，不要把串口缺失直接判断为 USB 网卡不能上网；
如果没有可用的 AT 通道，先检查 USB 枚举及串口支持，避免直接运行假定存在的 minicom 设备名。

常用状态和日志：

```bash
# USRP 主机
python3 ~/Open-RAN/lab/oai-gnb/scripts/labctl.py status
cd ~/oai-cn5g && docker compose ps
docker logs --tail 100 oai-gnb-next
systemctl status oai-link-monitor.service --no-pager
journalctl -u oai-link-monitor.service -n 40 --no-pager

# AGX
systemctl status oai-link-agent.service --no-pager
journalctl -u oai-link-agent.service -n 40 --no-pager
```

需要回退到保留的旧 gNB：

```bash
sudo python3 ~/Open-RAN/lab/oai-gnb/scripts/labctl.py restore
```

自定义本机目录时，启动入口支持 `--state-dir`、`--core-dir` 和 `--monitor-url`。
40 MHz 是该入口的已验证基准，其他带宽见[逐档实测记录](validation/bandwidth-2026-10-20/README.md)。
