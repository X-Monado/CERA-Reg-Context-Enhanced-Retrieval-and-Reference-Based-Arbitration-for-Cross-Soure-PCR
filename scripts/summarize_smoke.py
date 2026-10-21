"""Summarize a completed local smoke replay without changing reference results."""
from pathlib import Path
import argparse,json,csv
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
source=a.run/'acceptance.json'
if not source.exists():source=a.run/'results.json'
r=json.loads(source.read_text());assert r['complete']
with (a.run/'comparison.csv').open('w',newline='') as f:
 keys=['case','scene','seed','old_RRE','new_RRE','old_RTE_cm','new_RTE_cm','old_success','new_success','max_pose_delta','selected_match','seconds','peak_allocated_MiB','pass'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(r['rows'])
lines=['CERA-Reg 独立目录短时复现验收','', '环境：已有 geotransformer conda 环境；GPU 1，V100；无训练。','验证范围：原始点云读取 → 预处理 → 编码器 → 对应检索 → 候选拟合 → 仲裁 → 原坐标恢复。','模型代码仅从 CERA-Reg 目录导入；原项目只提供对照结果，不导入其实现。','权重和阈值不变，使用基准种子 7351 + 配对索引，与既有逐对评估一致。','对照：Cross_source_pcr/output/v100_completion_20261016/quality 对应 observed_7351 预测。','位姿矩阵最大元素差容限 1e-4；另检查成功判定与候选索引一致。','', '案例 | 原RRE/° | 新RRE/° | 原RTE/cm | 新RTE/cm | 原/新成功 | 位姿最大差 | 耗时/s | 峰值显存/MiB']
for x in r['rows']:lines.append(f"{x['case']} ({x['scene']}) | {x['old_RRE']:.4f} | {x['new_RRE']:.4f} | {x['old_RTE_cm']:.4f} | {x['new_RTE_cm']:.4f} | {x['old_success']}/{x['new_success']} | {x['max_pose_delta']:.3g} | {x['seconds']:.2f} | {x['peak_allocated_MiB']:.1f}")
lines+=['',f"已完成案例全部通过：{r['all_pass']}；两对推理累计 {sum(x['seconds'] for x in r['rows']):.2f} 秒（不含依赖导入与初始化）。",'时间用于本次资源预算估计，不作为算法速度比较；峰值为 PyTorch allocated，非进程全部显存。','这是2对SfM完整推理与2对CPU缓存回放的冒烟验收，不等于全量3DCSR成功率重算，也不等于新机器从零安装验收。','原计划4对GPU推理；按短测预算在完成2对SfM后主动中止，LiDAR原始推理未验收。保留原始中止日志与 complete=false 的 results.json。','GT按既有数据加载器先转float32再转float64复算；两例RRE/RTE也与归档完全一致。首次测试脚本直接以float64读取GT造成极小误差差异，不涉及模型位姿差异。','原有2对缓存的CPU评分器核验见上级 core_check.json，效用/风险最大差均为0，编码器缺失权重为0。','', '开源前待完善：独立训练入口与数据清单、干净环境编译安装、权重下载方式与许可证确认。','静态检查另发现旧训练辅助函数 d1_scene_split 保留个人数据路径；当前推理不调用它，正式整理训练入口时应改为显式数据根路径。','测试脚本首次误用了仅含对应质量字段的缓存进行评分器回放，已移除该不适用读取；未因此修改模型或结果。','启动等待经调用栈定位包含 Python/Matplotlib 依赖导入；另一张V100的CUDA算术检查通过，不能将启动耗时归因于驱动故障。']
(a.run/'复现验收报告.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
