问题来源：随着智能家居与物联网技术的不断发展，家庭电力消耗监测与管理成为节能减排、降低用 电成本和实现智能化生活的重要环节。通过对家庭电力消耗进行细致的数据采集和建模分析，不仅有 助于居民了解自身用电行为，还能为电力公司合理调度和预测电力负荷、平衡供需提供技术支持。家 庭电力消耗受多种因素影响，例如季节变换、节假日、家庭成员行为模式、用电设备种类及气象条件 等，这使得准确预测具有较大挑战性和现实意义。
对家庭电力消耗进行多变量时间序列预测，可以帮助用户及时发现异常用电，合理安排用电时间，降 低峰值负荷，从而节省电费和提升能源利用效率。同时，精确的用电预测对智能电网的动态调度、可 再生能源接入与分布式能源管理等也有重要推动作用。典型数据集为UCI Machine Learning  Repository 公 开 的 “ Individual household electric power consumption ” 数 据 集 ， 可 在 https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption 获 取，采集自法国一户家庭，记录时间跨度从2006年12月到2010年11月，数据粒度为每一分钟，包括全 屋有功功率、无功功率、电流、电压、各路子表的能耗等多个变量。我们通常会以每天为单位对原始 数据进行汇总，同时可融合天气等外部因素作为输入变量，构建多变量时间序列预测模型。数据以月 为 基 础 汇 总 数 据 ， 然 后 提 取 并 添 加 相 应 的 信 息 。 天 气 信 息 可 在 下 述 网 站 获 取 ： https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles。
数据各列含义如下： global_active_power（全局有功功率）：家庭总的有功功率消耗，单位为千瓦（kW），表示实际消 耗的电能。
global_reactive_power（全局无功功率）：家庭总的无功功率消耗，单位为千瓦（kW），表示储存 在电路中并来回转换的能量。 
voltage（电压）：电路中的平均电压值，单位为伏特（V）。 
global_intensity（电流强度）：家庭总的平均电流强度，单位为安培（A）。 
sub_metering_1（分表1能耗）：厨房区域的有功能量消耗，单位为瓦时（Wh）。 
sub_metering_2（分表2能耗）：洗衣房区域的有功能量消耗，单位为瓦时（Wh）。 
Sub_metering_3（分表3能耗）：气候控制系统的有功能量消耗，单位为瓦时（Wh）。 
RR：月累计降水高度，单位为毫米的十分之一（即记录值需除以10） 
NBJRR1 / NBJRR5 / NBJRR10 ：当月日降水 ≥ 1/5/10 mm的天数 
NBJBROU : 当月雾出现的天数 
提示：可以根据其他三个变量，计算出第四个变量：sub_metering_remainder =  (global_active_power * 1000 / 60) - (sub_metering_1 + sub_metering_2 + sub_metering_3) 
预测问题：根据最近的电力消耗情况，接下来的预期电力消耗是多少？即要求对接下来每一天的总有 功功率进行预测。 
预测任务：根据所提供的数据对未来总有功功率进行预测。基于过去90天的数据曲线来预测未来90天 （短期预测）和365天（长期预测）两种长度的变化曲线（需要分别训练，即长期预测的模型参数不 能用于短期预测）。按照方法分为三部分。 前两部分为基础题，第三部分为开放题，各占总分的三分之一： 
1. 使用 LSTM 模型进行预测；（对应code文件夹下LSTM_90.py与LSTM_365.py文件）
2. 使用 Transformer 模型进行预测；（对应code文件夹下Transformer_90.py与Transformer_365.py文件）
3. 使用自己提出的改进模型进行预测，结构不限，例如可以结合卷积层提取局部特征后接 Transformer编码以改进长期依赖建模能力。此部分以原理的新颖程度为首要评价标准，性能为次要评价标准。（对应code文件夹下RF-CNN-LSTM_90.py与RF-CNN-LSTM_365.py文件）
训练与测试： 数据集主要分为 train 和 test 两 部 分 （ 具体见文件“ train.csv ”和 “tes.csv”）。请使用两种评价标准进行测试，即均方误差（MSE）与平均绝对误差（MAE）。至少进行 五轮实验，并对结果取平均值，同时提供标准差（std）以评估结果的稳定性。 
提示：数据处理方面应该按照以下处理 global_active_power、global_reactive_power、sub_metering_1、sub_metering_2 按天取总和 voltage、global_intensity 按天取平均 RR、NBJRR1 / NBJRR5 / NBJRR10、NBJBROU 取当天的任意一个数据
