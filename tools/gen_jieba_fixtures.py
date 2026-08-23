#!/usr/bin/env python3
"""Generate the B5 jieba parity fixture corpus.

Runs the REAL golden tokenizer chain — jieba_fast.posseg.lcut(s), the exact
call CPUFast `text/chinese.py::_g2p` makes — over a >=200-sentence corpus and
writes expectations for the native C++ diff test.

Run with the GPTSoVits env python:
    /Users/baicai1145/miniconda3/envs/GPTSoVits/bin/python \
        tools/gen_jieba_fixtures.py --out tests/textfront/fixtures/jieba_fixtures.txt

Output format "GSVFIX01" (record-per-block):
    S\t<id>\t<text>          # text escaped: \\ -> \\\\, LF -> \\n, TAB -> \\t, CR -> \\r
    W\t<word>\t<flag>        # one line per expected token, same escaping
    <blank line>             # record terminator
"""
import argparse
import json
import os
import sys


def esc(s):
    return (s.replace("\\", "\\\\").replace("\n", "\\n")
             .replace("\t", "\\t").replace("\r", "\\r"))


def build_corpus(repo_root):
    corpus = []  # (category, sentence)

    def add(cat, *sents):
        for s in sents:
            corpus.append((cat, s))

    # --- golden manifest sentences (tests/golden/manifest.json) ----------
    manifest = os.path.join(repo_root, "tests", "golden", "manifest.json")
    with open(manifest, encoding="utf-8") as f:
        golden = json.load(f)["sentences"]
    add("golden", *golden)
    add("golden", "原来你也玩原神。")

    # --- pure Chinese -----------------------------------------------------
    add("pure_zh",
        "今天星期三。", "明天可能会下雨。", "他正在图书馆看书。",
        "这家餐厅的菜味道不错，服务也很好。", "春天来了，花都开了。",
        "小明每天早上六点起床跑步。", "中国的首都是北京。",
        "长江是中国最长的河流。", "这本书讲的是中国古代历史。",
        "医生建议他多休息，少熬夜。", "孩子们在操场上踢足球。",
        "妈妈做红烧肉特别好吃。", "火车马上就要进站了。",
        "会议改到下午三点召开。", "快递员把包裹放在了门口。",
        "这幅画的色彩非常鲜艳。", "爷爷喜欢在公园里下象棋。",
        "天气预报说周末有台风。", "她穿了一件红色的连衣裙。",
        "电脑突然死机了，文件还没保存。", "老师表扬了全班同学。",
        "小区门口新开了一家超市。", "这道数学题太难了，我不会做。",
        "大熊猫是中国的国宝。", "高铁从上海到北京只要四个半小时。",
        "他把钥匙忘在办公室了。", "夜空中的星星一闪一闪的。",
        "这个手机电池不太耐用。", "奶奶种的西瓜又大又甜。",
        "图书馆借书要按时归还。", "运动员们正在进行赛前热身。",
        "博物馆里陈列着许多文物。", "弟弟的梦想是当一名飞行员。",
        "厨房里飘来阵阵香味。", "公司年会定在下个月举行。",
        "海水退潮后露出大片沙滩。", "他修好了那台旧收音机。",
        "山上的寺庙有一百多年历史。", "候车室里人山人海。",
        "这批货物明天早上装船。", "导演正在给演员讲戏。",
        "果园里的苹果熟透了。", "地震预警系统发挥了作用。",
        "邻居家的猫又跑丢了。", "快递柜满了只能放门口。",
        "新来的同事很热心。", "电梯里的广告太多了。")

    # --- numbers / dates / measures ----------------------------------------
    add("numbers",
        "2024年10月1日，我在成都买了3.5斤樱桃。",
        "我的电话号码是13812345678。",
        "这次考试他得了98.5分。",
        "会议室能容纳120个人。",
        "房价每平方米50000元。",
        "飞机于13时25分起飞。",
        "他身高一米八五，体重75公斤。",
        "银行利率下调了0.25个百分点。",
        "项目预算总计1000万元人民币。",
        "第29届奥运会于2008年举办。",
        "保质期还剩6个月零8天。",
        "车速限制每小时60公里。",
        "文件大小是2.5GB。",
        "气温降到了零下10度。",
        "订单编号20240931001已发货。",
        "折扣力度是七五折。",
        "存款利息按3.25%计算。",   # '%'不在posseg re_han内，走非匹配块
        "增长率达到15.8%。", "浓度是0.9%的氯化钠溶液。",
        "他买了5斤苹果、3斤橘子和2斤香蕉。",
        "1949年10月1日新中国成立。",
        "圆周率约等于3.14159。",
        "比赛最终比分是3比2。",
        "这栋楼有33层，高108米。",
        "二维码扫描成功率超过99.9%。",
        "他的车牌号是京A12345。", "航班CA1856延误两小时。",
        "经纬度为北纬39.9度、东经116.4度。")

    # --- English / alphanumeric mixing --------------------------------------
    add("english",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-SoVITS 是一个开源的语音合成项目。",
        "我喜欢用Python写脚本，也用C++写性能关键的代码。",
        "请把文件保存为PDF格式发给我。",
        "他的邮箱是test_user@example.com。",
        "iPhone 15 Pro Max售价999美元。",
        "WiFi密码是abcd1234。",
        "AI技术正在快速发展。",
        "这个API支持JSON和XML格式。",
        "CPU温度达到了85度。",
        "Hello World！你好世界！",
        "登录https://example.com查看详情。",
        "DNA检测报告显示一切正常。",
        "KPI考核标准已经更新了。",
        "用SSH连接到服务器192.168.1.100。",
        "GDP同比增长5.2%。",
        "App Store上有几百万个应用。",
        "USB接口松动了，接触不良。")

    # --- fullwidth punctuation / symbols ------------------------------------
    add("fullwidth",
        "你好！很高兴见到你？",
        "《红楼梦》是四大名著之一。",
        "他说：「今天天气真好。」",
        "（括号里的内容仅供参考。）",
        "第一、准备工作；第二、制定计划；第三、执行。",
        "哇～这里的风景太美了！",
        "……我不知道该说什么好。",
        "甲——乙——丙——丁。",
        "【重要通知】明日停水检修。",
        "数字１２３４５和全角ＡＢＣ。",
        "疑问句：这是真的吗？？？",
        "感叹！！太厉害了！！！",
        "书名号《三体》、引号“人工智能”、顿号、逗号。",
        "波浪线～和破折号——还有省略号……。",
        "中文分号；英文分号;混合使用。",
        "全角冒号：半角冒号:对比。",
        "「直接引语」与『嵌套引语』的区别。",
        "百分号％是全角的，%是半角的。",
        "加减乘除：＋－×÷。",
        "箭头符号→←↑↓表示方向。")

    # --- long sentences (>=100 chars) ---------------------------------------
    add("long",
        "人工智能技术在过去几年里取得了令人瞩目的进展，特别是在自然语言处理、计算机视觉和语音识别等领域，"
        "各种大规模预训练模型的出现极大地推动了相关产业的发展，同时也引发了人们对于数据隐私、算法偏见和技术伦理"
        "等问题的广泛讨论，如何在推动技术进步的同时确保其安全可靠地服务于人类社会，已经成为整个行业必须认真思考"
        "的重要课题。",
        "中国的饮食文化源远流长，八大菜系各具特色：川菜麻辣鲜香，粤菜清淡鲜美，鲁菜醇厚浓郁，苏菜精细雅致，"
        "浙菜清爽细腻，闽菜清鲜和醇，湘菜香辣酸辣，徽菜重油重色，每一種菜系背后都蕴含着当地的历史传统与风土人情，"
        "无论是街边小摊还是高档餐厅，都能让人感受到中华美食的博大精深与无穷魅力，这也正是中国文化的独特之处。",
        "小明放学回到家，放下书包就迫不及待地打开电视，因为他最喜欢的动画片马上就要开始了，可是妈妈却让他先完成"
        "作业再玩，小明虽然有些不情愿，但还是乖乖地坐到书桌前，认认真真地写起了作业，等到全部做完的时候，动画片"
        "已经播完了，不过妈妈答应他周末带他去游乐园玩一整天，这让小明的心情一下子又好了起来。",
        "这座城市的交通网络四通八达，地铁线路纵横交错，公交车穿梭于大街小巷，共享单车解决了最后一公里的出行难题，"
        "网约车提供了更加个性化的选择，随着智慧城市建设的不断推进，交通信号灯变得更加智能，道路通行效率显著提升，"
        "市民出行的体验越来越好，城市也变得越来越宜居。")

    # --- HMM sensitive: names/places/orgs/rare words -------------------------
    add("hmm",
        "张伟和李娜都是北京人。",
        "刘德华在台北举办了演唱会。",
        "诸葛亮是三国时期蜀汉的丞相。",
        "乌鲁木齐是新疆的首府。",
        "呼和浩特位于内蒙古自治区。",
        "马尔代夫是著名的旅游胜地。",
        "秦始皇统一了六国。",
        "李白和杜甫是唐代著名诗人。",
        "马云创立了阿里巴巴集团。",
        "清华大学的校园很漂亮。",
        "长江三峡水利枢纽工程宏伟壮观。",
        "喜马拉雅山脉是世界屋脊。",
        "布达拉宫矗立在拉萨河边。",
        "成吉思汗建立了蒙古帝国。",
        "莎士比亚写了《哈姆雷特》。",
        "爱因斯坦提出了相对论。",
        "螭吻和饕餮都是上古神兽。",
        "貔貅有招财纳福的寓意。",
        "耄耋之年的老人依然精神矍铄。",
        "魑魅魍魉泛指各种鬼怪。",
        "彳亍不前的行人。", "孑孓在水里游动。",
        "囹圄之中仍不忘初衷。", "龌龊的行为令人不齿。",
        "犹豫不决会错失良机。", "忐忑不安的心情。",
        "尴尬的气氛弥漫开来。", "唠叨是爱的另一种表达。",
        "葡萄美酒夜光杯。", "琉璃瓦在阳光下闪闪发光。",
        "咖啡厅里放着爵士乐。", "沙发上的抱枕很柔软。",
        "幽默感是社交的润滑剂。", "逻辑思维需要刻意练习。",
        "模特走了猫步。", "沙龙聚会持续到深夜。",
        "敦煌壁画色彩斑斓。", "泉州是宋元时期的东方大港。",
        "景德镇瓷器享誉世界。", "茅台酒产于贵州仁怀。",
        "普洱茶越陈越香。", "苏州园林精巧雅致。",
        "鼓浪屿的小巷幽深静谧。")

    # --- whitespace / control characters -------------------------------------
    add("whitespace",
        "hello world", "a b c d", "行一分\n行二分",
        "tab\t分隔符", "回车符\r\n换行测试", "全角空格\u3000测试",
        "不间断空格\xa0测试", "前导空格   前导", "尾随空格   ",
        "多个   连续   空格", "混合 \t 空白\n字符 测试", "x")

    # --- tricky mixed ---------------------------------------------------------
    add("tricky",
        "",  # 空串
        "。", "！", "，。？！；：、", "的的的的", "了了了了了",
        "A+B=C&D.e_f", "IPv6地址是2001:db8::1",
        "😀表情符号测试🎉", "中文English日本語にほんご한국어",
        "@#$%^&*()", "～～～～", "。。。。。。", "！！！！",
        "e", "E", "3", ".", "_", "#标签 @提及 http://t.cn/abc",
        "第1章 第2节 第3部分",
        "v2ProPlus模型包含enc_p、flow、dec三个模块。",
        "①②③④⑤序号字符。", "℃温度符号和℉。")

    return corpus


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out",
                    default=os.path.join(here, "tests", "textfront",
                                         "fixtures", "jieba_fixtures.txt"))
    args = ap.parse_args()

    sys.path.insert(0,
        "/Users/baicai1145/miniconda3/envs/GPTSoVits/lib/python3.10/site-packages")
    import jieba_fast
    jieba_fast.setLogLevel(50)
    import jieba_fast.posseg as psg

    corpus = build_corpus(here)

    # dedupe by text, keep first occurrence
    seen = set()
    uniq = []
    for cat, s in corpus:
        if s not in seen:
            seen.add(s)
            uniq.append((cat, s))
    print(f"corpus: {len(corpus)} entries, {len(uniq)} unique sentences")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_tokens = 0
    stats = {}
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("GSVFIX01\n")
        for idx, (cat, text) in enumerate(uniq):
            toks = [(w.word, w.flag) for w in psg.lcut(text)]
            n_tokens += len(toks)
            stats[cat] = stats.get(cat, 0) + 1
            f.write(f"S\t{idx:04d}_{cat}\t{esc(text)}\n")
            for w, flag in toks:
                f.write(f"W\t{esc(w)}\t{flag}\n")
            f.write("\n")
    print(f"wrote {args.out}: {len(uniq)} cases, {n_tokens} tokens")
    print("categories:", dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
