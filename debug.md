# Debug Session: 20260322-1
- **Status**: [OPEN]
- **Issue**: GUI运行卡顿，控制板模式启动仍疑似触发硬件边界读取

## Reproduction Steps (Repro Steps)
1. 打开GUI并进入控制板模式
2. 开始运行/跟踪
3. 观察卡顿现象

## Hypotheses & Verification (Hypotheses)
- [ ] Hypothesis A: 采集线程单帧耗时过长导致UI阻塞 | Evidence: Pending
- [ ] Hypothesis B: 检测线程循环过密，CPU占满导致卡顿 | Evidence: Pending
- [ ] Hypothesis C: 队列阻塞或积压导致UI线程等待 | Evidence: Pending
- [ ] Hypothesis D: 控制板模式仍触发硬件边界读取或串口阻塞 | Evidence: Pending

## Verification Conclusion (Verification)
Pending pre-fix logs
