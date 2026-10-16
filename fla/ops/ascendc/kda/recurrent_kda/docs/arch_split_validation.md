# RecurrentKda 架构拆分验证记录

当前状态：架构拆分及指定精度验证已完成，性能尚未全部达到验收条件。

## 改动范围

- host 和 kernel 分别维护 arch22（A2/A3）与 arch35（A5）实现，各架构的 processor、tiling data 和命名空间独立，架构目录不交叉包含实现。
- 公共 host 入口仅注册一次并按架构分发；公共 kernel 入口按编译架构选择实现。公开 API、参数顺序、计算公式及 tiling data 的 256 字节布局保持不变。
- 修正长期缓存 FetchEventID 返回值的资源生命周期问题。arch22 的队列相关事件仅用于即时 set/wait，V_S 独立占用；arch35 的缓存事件使用 AllocEventID 占用，并逐项释放。
- arch22 以余数递推和 head 步进替代逐 head 扫描。任务归属、顺序、空序列处理和 state 预取顺序保持一致；静态枚举核对覆盖 21,420 种核配置及 372,708 个序列列表。
- 未采用 beta 缓存、向量 sigmoid 或 Q/K normalize 原型。

## 构建和精度

- 按 README 仅构建、安装 recurrent_kda 的 ascend910b 包；构建通过，72 个安装对象与构建包逐项一致，实际加载的 OP API 库及性能采样 kernel 已校验。
- 四个 base case（ID 0、2、4、6；B=1、4、16、64）使用默认 Stable ABI 路径，精度 4/4 通过；输出和更新后的 state 也通过 CPU 参考对比。
- 补充 ATK 200 个 case 使用原用例、原标杆和原精度标准，精度 200/200 通过。该组使用公开 ctypes/aclnn 路径（FLA_NPU_STABLE_ABI=ctypes）；默认 Stable ABI 适配层既有的非 inplace／缺省 initial_state 限制未在本次改动中调整。
- arch35 代表性 Ascend950 编译通过；未完成 A5 设备运行验收。

## 性能

使用 msopprof BasicInfo 的 kernel Task Duration，执行两轮交错顺序比较；每轮每版 30 个样本，表中为每版全部 60 个样本的合并中位数。未使用 Python wall time。

| B | 起始实现（us） | 重构版本（us） | 耗时变化 | 不劣化 |
| --- | ---: | ---: | ---: | --- |
| 1 | 31.83 | 33.73 | +5.97% | 未通过 |
| 4 | 83.41 | 82.28 | -1.35% | 通过 |
| 16 | 286.70 | 273.78 | -4.51% | 通过 |
| 64 | 1089.75 | 1029.01 | -5.57% | 通过 |

B=1 回退，整体性能验收未通过。起始实现的四个 base case 会产生 NaN，精度为 0/4；以上计时只反映与该起始实现的耗时差异，不能把它描述为精度合格的性能基线。后续需使用精度通过的基线包或提交进一步核验性能。

## 内存检查

当前源码按 `--op_debug_level=1 --op_debug_config=dump_cce,sanitizer` 及 Ascend C sanitizer 选项编译。已通过 nm 符号和 `Start racecheck sanitizer on kernel` 日志确认插桩对象实际命中，但单 case 运行发生 507035 向量核异常，程序异常退出，未获得可通过的内存检查结论；根因尚未确定。起始版本的插桩运行也出现同类异常。运行后已恢复正常包，并核对全部 72 个 kernel 对象及配置与正常构建产物一致。内存检查结果独立于精度与性能结果，不以工具包装进程退出码代替。
