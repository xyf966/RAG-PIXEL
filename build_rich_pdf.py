from pathlib import Path
import json
import math

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rag_multimodal_testset" / "rich_pdf"
ASSETS = OUT / "assets"
PDF_PATH = OUT / "05_天枢冷链中心多模态运营分析报告.pdf"
QA_PATH = OUT / "05_天枢冷链中心_ground_truth.jsonl"

NAVY = "#17324d"
BLUE = "#2f6690"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GOLD = "#e9c46a"
LIGHT = "#eef4f7"
GRID = "#d9e3e8"


def font(size, bold=False):
    paths = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for p in paths:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def centered(draw, xy, text, f, fill=NAVY):
    box = draw.textbbox((0, 0), text, font=f)
    draw.text((xy[0] - (box[2]-box[0])/2, xy[1] - (box[3]-box[1])/2), text, font=f, fill=fill)


def base_canvas(title, subtitle=None):
    im = Image.new("RGB", (1400, 820), "white")
    d = ImageDraw.Draw(im)
    centered(d, (700, 45), title, font(36, True))
    if subtitle:
        centered(d, (700, 88), subtitle, font(21), "#526772")
    return im, d


def make_scene(path):
    im = Image.new("RGB", (1500, 900), "#e4edf2")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 570, 1500, 900), fill="#c5d2d9")
    d.line((0, 570, 1500, 570), fill="#607d8b", width=6)
    for x in range(0, 1501, 150): d.line((750, 570, x, 900), fill="#a6b8c1", width=3)
    # Left wall: warning sign and extinguisher.
    d.polygon([(110, 100), (40, 220), (180, 220)], fill="#ffd600", outline="#7a6300")
    centered(d, (110, 180), "!", font(62, True), "#272000")
    d.rounded_rectangle((65, 330, 145, 520), radius=18, fill="#d32f2f", outline="#6f1111", width=6)
    d.arc((70, 270, 165, 390), 180, 350, fill="#333333", width=10)
    d.rectangle((90, 300, 122, 340), fill="#333333")
    # Blue robotic arm.
    d.rectangle((480, 600, 710, 715), fill="#1565c0", outline="#073662", width=7)
    d.ellipse((525, 455, 660, 590), fill="#1976d2", outline="#073662", width=7)
    d.line((590, 515, 730, 340), fill="#1976d2", width=58)
    d.ellipse((695, 300, 790, 395), fill="#1976d2", outline="#073662", width=7)
    d.line((742, 345, 850, 470), fill="#1976d2", width=45)
    d.line((840, 455, 900, 420), fill="#263238", width=16)
    d.line((840, 455, 900, 490), fill="#263238", width=16)
    # Female engineer: white helmet, orange vest, black tablet.
    d.ellipse((275, 265, 370, 365), fill="#efb487", outline="#6d4c41", width=4)
    d.pieslice((265, 230, 380, 335), 180, 360, fill="white", outline="#607d8b", width=5)
    d.polygon([(295, 365), (360, 365), (405, 585), (250, 585)], fill="#f57c00", outline="#7a3d00")
    d.line((275, 395, 205, 510), fill="#6d4c41", width=20)
    d.line((360, 400, 430, 500), fill="#6d4c41", width=20)
    d.rounded_rectangle((390, 425, 490, 555), radius=8, fill="#20272b", outline="#90caf9", width=5)
    d.line((290, 585, 280, 735), fill="#263238", width=30); d.line((365, 585, 390, 735), fill="#263238", width=30)
    # Exactly two green AMRs.
    for x in (890, 1105):
        d.rounded_rectangle((x, 600, x+175, 760), radius=35, fill="#2eaa62", outline="#14502e", width=7)
        d.rectangle((x+40, 635, x+135, 690), fill="#c9f6d9", outline="#14502e", width=4)
        d.ellipse((x+25, 735, x+70, 780), fill="#263238"); d.ellipse((x+105, 735, x+150, 780), fill="#263238")
    # Exactly three silver insulated containers, rear-right.
    for i, x in enumerate((980, 1145, 1310)):
        d.rectangle((x, 345, x+135, 555), fill="#b0bec5", outline="#455a64", width=6)
        d.rectangle((x+15, 365, x+120, 535), outline="#eef5f7", width=5)
        d.line((x+67, 365, x+67, 535), fill="#607d8b", width=3)
    # Clock with only requested digits.
    d.rounded_rectangle((650, 75, 850, 175), radius=15, fill="#15232b", outline="#78909c", width=5)
    centered(d, (750, 125), "08:30", font(48, True), "#7cff8b")
    centered(d, (750, 35), "冷链智能巡检现场", font(36, True))
    im.save(path, quality=95)


def make_line_chart(path):
    months = [f"{i}月" for i in range(1, 13)]
    series = {
        "准时交付率": ([97.2,97.8,98.3,98.6,98.9,99.1,99.0,99.2,99.3,99.4,99.5,99.6], BLUE),
        "设备可用率": ([98.6,98.8,98.9,98.4,99.0,99.2,99.1,99.3,99.4,99.2,99.6,99.5], TEAL),
        "冷链合规率": ([99.1,98.7,98.2,96.8,97.6,98.4,98.8,99.0,99.2,99.1,99.3,99.4], ORANGE),
    }
    im, d = base_canvas("2025 年月度服务质量趋势", "三条折线共用百分比纵轴；虚线为 98.0% 目标线")
    L,T,R,B = 110,135,1330,690; ymin,ymax = 96,100
    for v in [96,97,98,99,100]:
        y = B-(v-ymin)/(ymax-ymin)*(B-T); d.line((L,y,R,y), fill=GRID, width=2); d.text((28,y-14), f"{v}%", font=font(22), fill="#526772")
    target_y = B-(98-ymin)/(ymax-ymin)*(B-T)
    for x in range(L,R,20): d.line((x,target_y,min(x+10,R),target_y), fill="#6b7280", width=4)
    d.text((R-155,target_y-30), "目标 98.0%", font=font(21, True), fill="#4b5563")
    step=(R-L)/11
    for i,m in enumerate(months): centered(d,(L+i*step,B+35),m,font(18),"#526772")
    for name,(vals,col) in series.items():
        pts=[]
        for i,v in enumerate(vals): pts.append((L+i*step, B-(v-ymin)/(ymax-ymin)*(B-T)))
        d.line(pts, fill=col, width=6)
        for p in pts: d.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6), fill=col, outline="white", width=2)
    # Legend and explicit exception annotation.
    x=260
    for name,(_,col) in series.items():
        d.line((x,755,x+45,755),fill=col,width=7); d.text((x+55,738),name,font=font(20),fill=NAVY); x+=330
    apr_x=L+3*step; apr_y=B-(96.8-ymin)/(ymax-ymin)*(B-T)
    d.line((apr_x,apr_y,apr_x+80,apr_y-75), fill=ORANGE, width=3)
    d.rounded_rectangle((apr_x+75,apr_y-115,apr_x+285,apr_y-60), radius=8, fill="#fff0eb", outline=ORANGE, width=2)
    d.text((apr_x+88,apr_y-105), "4月：96.8%", font=font(21,True), fill=ORANGE)
    im.save(path, quality=95)


def make_combo_chart(path):
    months=[f"{i}月" for i in range(1,13)]
    orders=[120,132,148,155,167,176,182,190,198,210,218,225]
    energy=[8.4,8.2,8.0,7.9,7.7,7.5,7.3,7.1,6.9,6.7,6.4,6.2]
    im,d=base_canvas("订单量与单位能耗双轴组合图", "蓝色柱：订单量（千单，左轴）｜橙色折线：单位能耗（kWh/单，右轴）")
    L,T,R,B=120,145,1280,690
    for v in range(0,251,50):
        y=B-v/250*(B-T); d.line((L,y,R,y),fill=GRID,width=2); d.text((42,y-13),str(v),font=font(20),fill="#526772")
    for v in [6,7,8,9]:
        y=B-(v-5.5)/(9-5.5)*(B-T); d.text((1290,y-13),f"{v:.1f}",font=font(20),fill=ORANGE)
    step=(R-L)/12; bw=step*.55; pts=[]
    for i,(m,o,e) in enumerate(zip(months,orders,energy)):
        cx=L+step*(i+.5); y=B-o/250*(B-T)
        d.rounded_rectangle((cx-bw/2,y,cx+bw/2,B),radius=6,fill=BLUE)
        centered(d,(cx,B+31),m,font(17),"#526772")
        ey=B-(e-5.5)/(9-5.5)*(B-T); pts.append((cx,ey))
    d.line(pts,fill=ORANGE,width=7)
    for p,e in zip(pts,energy): d.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill=ORANGE,outline="white",width=2)
    d.text((L,720),"1月 120千单 / 8.4",font=font(19,True),fill=NAVY)
    d.text((R-245,720),"12月 225千单 / 6.2",font=font(19,True),fill=NAVY)
    im.save(path,quality=95)


def make_stacked(path):
    zones=["A区","B区","C区","D区"]
    data={"拣选错误":[12,9,14,6],"包装破损":[8,11,6,5],"温控偏差":[3,7,9,2]}
    palette={"拣选错误":BLUE,"包装破损":GOLD,"温控偏差":ORANGE}
    im,d=base_canvas("各作业区异常类型构成", "堆叠柱状图；数字为异常事件数")
    L,T,R,B=125,140,1280,690; ymax=35
    for v in range(0,36,5):
        y=B-v/ymax*(B-T); d.line((L,y,R,y),fill=GRID,width=2); d.text((60,y-12),str(v),font=font(20),fill="#526772")
    step=(R-L)/4; bw=150
    for i,z in enumerate(zones):
        cx=L+step*(i+.5); cum=0
        for name,vals in data.items():
            val=vals[i]; y1=B-cum/ymax*(B-T); cum+=val; y2=B-cum/ymax*(B-T)
            d.rectangle((cx-bw/2,y2,cx+bw/2,y1),fill=palette[name],outline="white",width=3)
            centered(d,(cx,(y1+y2)/2),str(val),font(23,True),"white" if name!= "包装破损" else NAVY)
        centered(d,(cx,B+35),z,font(23,True)); centered(d,(cx,B-cum/ymax*(B-T)-28),f"合计{cum}",font(19,True))
    x=300
    for name in data:
        d.rectangle((x,750,x+28,778),fill=palette[name]); d.text((x+40,748),name,font=font(19),fill=NAVY); x+=290
    im.save(path,quality=95)


def make_donut(path):
    labels=[("机械故障",35,BLUE),("软件异常",25,TEAL),("供电问题",20,GOLD),("网络中断",12,ORANGE),("其他",8,"#8d99ae")]
    im,d=base_canvas("停机原因占比", "全年停机总时长：400 小时")
    box=(180,145,760,725); start=-90
    for name,pct,col in labels:
        end=start+pct/100*360; d.pieslice(box,start,end,fill=col,outline="white",width=5); start=end
    d.ellipse((350,315,590,555),fill="white")
    centered(d,(470,405),"400",font(58,True)); centered(d,(470,470),"小时",font(28),"#526772")
    y=190
    for name,pct,col in labels:
        d.rectangle((870,y,910,y+40),fill=col); d.text((930,y+3),f"{name}  {pct}%",font=font(25,True),fill=NAVY); y+=90
    d.rounded_rectangle((860,655,1280,725),radius=12,fill="#eef4f7",outline="#b8c8d0")
    d.text((885,673),"机械故障：140 小时",font=font(24,True),fill=BLUE)
    im.save(path,quality=95)


def make_heatmap(path):
    vals=[[-19.2,-19.0,-18.8,-18.5,-18.6,-18.9],[-19.0,-18.7,-18.4,-18.2,-17.9,-18.4],[-18.8,-18.5,-18.0,-17.6,-17.2,-17.8],[-18.7,-18.3,-17.8,-17.0,-16.4,-17.2],[-18.9,-18.4,-17.9,-16.8,-14.2,-16.9],[-19.1,-18.6,-18.2,-17.7,-17.3,-18.0]]
    im,d=base_canvas("冷库温度传感器热力图", "行 A–F，列 1–6；安全阈值 ≤ -18.0°C")
    x0,y0,cw,ch=250,150,145,85
    def col(v):
        t=max(0,min(1,(v+20)/6)); return (int(40+210*t),int(130-70*t),int(190-120*t))
    for r,row in enumerate(vals):
        centered(d,(190,y0+r*ch+ch/2),chr(65+r),font(25,True))
        for c,v in enumerate(row):
            d.rectangle((x0+c*cw,y0+r*ch,x0+(c+1)*cw,y0+(r+1)*ch),fill=col(v),outline="white",width=4)
            centered(d,(x0+c*cw+cw/2,y0+r*ch+ch/2),f"{v:.1f}",font(23,True),"white")
    for c in range(6): centered(d,(x0+c*cw+cw/2,120),str(c+1),font(24,True))
    d.rectangle((x0+4*cw,y0+4*ch,x0+5*cw,y0+5*ch),outline="#ffeb3b",width=9)
    d.text((1120,525),"热点：E5",font=font(25,True),fill=ORANGE); d.text((1120,570),"-14.2°C",font=font(30,True),fill=ORANGE)
    im.save(path,quality=95)


def make_floorplan(path):
    im,d=base_canvas("天枢冷链中心平面示意图", "箭头表示夜间补货路线")
    zones=[((90,150,430,380),"A区\n收货",BLUE),((490,150,830,380),"B区\n冷藏",TEAL),((890,150,1230,380),"C区\n冷冻","#577590"),
           ((90,455,430,685),"D区\n分拣",GOLD),((490,455,830,685),"E区\n包装",ORANGE),((890,455,1230,685),"F区\n发货","#8d99ae")]
    for box,label,col in zones:
        d.rounded_rectangle(box,radius=22,fill=col,outline=NAVY,width=5); centered(d,((box[0]+box[2])/2,(box[1]+box[3])/2),label,font(33,True),"white")
    route=[(250,395),(660,395),(660,430),(1060,430)]
    d.line(route,fill="#d90429",width=14,joint="curve")
    d.polygon([(1060,430),(1015,405),(1015,455)],fill="#d90429")
    d.rounded_rectangle((1130,710,1320,780),radius=10,fill="#35a853"); centered(d,(1225,745),"东门集合点",font(23,True),"white")
    im.save(path,quality=95)


def make_flow(path):
    im,d=base_canvas("一级事件响应流程", "节点编号用于跨图检索")
    labels=[("S1", "发现并上报", BLUE),("S2", "5分钟内确认", TEAL),("S3", "15分钟首轮处置", GOLD),("S4", "30分钟升级", ORANGE),("S5", "复盘归档", "#8d99ae")]
    xs=[120,375,630,885,1140]
    for i,((code,text,col),x) in enumerate(zip(labels,xs)):
        d.rounded_rectangle((x,270,x+190,500),radius=25,fill=col,outline=NAVY,width=5)
        centered(d,(x+95,330),code,font(36,True),"white"); centered(d,(x+95,415),text,font(24,True),"white")
        if i<4:
            d.line((x+190,385,xs[i+1]-15,385),fill=NAVY,width=8); d.polygon([(xs[i+1]-15,385),(xs[i+1]-45,368),(xs[i+1]-45,402)],fill=NAVY)
    im.save(path,quality=95)


def cn_styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    base=getSampleStyleSheet()
    return {
        "title":ParagraphStyle("T",parent=base["Title"],fontName="STSong-Light",fontSize=25,leading=34,textColor=colors.HexColor(NAVY),alignment=TA_CENTER),
        "h1":ParagraphStyle("H1",parent=base["Heading1"],fontName="STSong-Light",fontSize=18,leading=25,textColor=colors.HexColor(BLUE),spaceAfter=8),
        "h2":ParagraphStyle("H2",parent=base["Heading2"],fontName="STSong-Light",fontSize=13,leading=19,textColor=colors.HexColor(NAVY),spaceAfter=5),
        "body":ParagraphStyle("B",parent=base["BodyText"],fontName="STSong-Light",fontSize=10.5,leading=17,textColor=colors.HexColor("#263238")),
        "cap":ParagraphStyle("C",parent=base["BodyText"],fontName="STSong-Light",fontSize=9.5,leading=15,textColor=colors.HexColor("#526772"),alignment=TA_CENTER),
        "small":ParagraphStyle("S",parent=base["BodyText"],fontName="STSong-Light",fontSize=8.7,leading=13,textColor=colors.HexColor("#526772")),
    }


def table(data, widths, header=True, font_size=9.5, row_bgs=True):
    t=Table(data,colWidths=widths,repeatRows=1 if header else 0)
    cmds=[("FONTNAME",(0,0),(-1,-1),"STSong-Light"),("FONTSIZE",(0,0),(-1,-1),font_size),("GRID",(0,0),(-1,-1),.45,colors.HexColor("#9fb2bc")),
          ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]
    if header: cmds += [("BACKGROUND",(0,0),(-1,0),colors.HexColor(BLUE)),("TEXTCOLOR",(0,0),(-1,0),colors.white)]
    if row_bgs: cmds += [("ROWBACKGROUNDS",(0,1 if header else 0),(-1,-1),[colors.white,colors.HexColor(LIGHT)])]
    t.setStyle(TableStyle(cmds)); return t


def page_footer(canvas, doc):
    canvas.saveState(); canvas.setFont("STSong-Light",8); canvas.setFillColor(colors.HexColor("#607d8b"))
    canvas.drawString(18*mm,10*mm,"NEBULA-PDF-RICH-2025-05｜仅用于多模态 RAG 基准测试")
    canvas.drawRightString(192*mm,10*mm,f"第 {doc.page} 页")
    canvas.restoreState()


def build_pdf(assets):
    s=cn_styles()
    doc=SimpleDocTemplate(str(PDF_PATH),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,bottomMargin=16*mm,
                          title="天枢冷链中心 2025 多模态运营分析报告",author="RAG Benchmark",subject="复杂图表、图片、表格和跨页检索测试")
    st=[]
    # Page 1 cover.
    st += [Spacer(1,10*mm),Paragraph("天枢冷链中心 2025<br/>多模态运营分析报告",s["title"]),Spacer(1,6*mm),
           Paragraph("复杂 PDF 输入基准｜版本 1.0｜2026 年 1 月发布",s["cap"]),Spacer(1,8*mm),
           RLImage(str(assets["scene"]),width=174*mm,height=104.4*mm),Spacer(1,5*mm),
           table([["文档编号","保密级别","报告负责人"],["NEBULA-PDF-RICH-2025-05","内部测试","苏衡"]],[58*mm]*3,font_size=10),
           Spacer(1,5*mm),Paragraph("报告范围：运营质量、能耗效率、异常分布、冷库温控、应急流程与空间布局。",s["body"]),PageBreak()]
    # Page 2 executive overview.
    st += [Paragraph("1. 执行摘要与关键指标",s["h1"]),
           Paragraph("天枢冷链中心位于南京江宁区，项目代号“北斗”。中心于 2025 年 2 月 17 日投入正式运营，运营负责人为苏衡，冷链质量负责人为叶岚。年度核心目标包括：准时交付率不低于 98.0%，单位能耗低于 6.5 kWh/单，重大安全事故为零。",s["body"]),Spacer(1,4*mm),
           table([["指标","2024基线","2025结果","目标","结论"],["全年订单量","162万单","212万单","≥200万单","达标"],["准时交付率","96.9%","99.6%","≥98.0%","达标"],
                  ["单位能耗","8.7 kWh/单","6.2 kWh/单","<6.5 kWh/单","达标"],["冷链合规率","97.8%","99.4%","≥98.0%","达标"],["重大安全事故","1起","0起","0起","达标"]],
                 [42*mm,33*mm,38*mm,33*mm,28*mm],font_size=9.2),Spacer(1,5*mm),
           Paragraph("关键解释",s["h2"]),Paragraph("订单量增长主要来自医药冷链业务；4 月冷链合规率短暂跌至 96.8%，原因是 C 区除霜控制器参数漂移。修复于 4 月 19 日完成，后续月份恢复到目标线以上。",s["body"]),Spacer(1,4*mm),
           Paragraph("下一次经营复盘安排在 2026 年 1 月 16 日 14:00，地点为玄武厅。",s["body"]),PageBreak()]
    # Page 3 scene.
    st += [Paragraph("2. 现场视觉巡检",s["h1"]),RLImage(str(assets["scene"]),width=174*mm,height=104.4*mm),
           Paragraph("图 1　冷链智能巡检现场。答案应基于图片本身，而非附近文字推测。",s["cap"]),Spacer(1,5*mm),
           Paragraph("视觉检查项",s["h2"]),table([["检查对象","规则"],["个人防护","进入设备区必须佩戴安全帽与反光背心"],["消防设施","灭火器不得被货物遮挡"],["机器人通道","移动机器人前方保持 1.2 米净空"],["保温货箱","外壳破损面积不得超过 10 cm²"]],[55*mm,119*mm],font_size=9.5),PageBreak()]
    # Page 4 line chart.
    st += [Paragraph("3. 服务质量月度趋势",s["h1"]),RLImage(str(assets["line"]),width=174*mm,height=101.9*mm),
           Paragraph("图 2　多系列折线图：准时交付率、设备可用率和冷链合规率。",s["cap"]),Spacer(1,4*mm),
           Paragraph("解读：4 月仅冷链合规率低于 98.0% 目标线；11 月设备可用率达到全年峰值 99.6%；12 月准时交付率为 99.6%。",s["body"]),PageBreak()]
    # Page 5 combo chart.
    st += [Paragraph("4. 订单规模与能效",s["h1"]),RLImage(str(assets["combo"]),width=174*mm,height=101.9*mm),
           Paragraph("图 3　双轴组合图：订单量逐月增加，单位能耗逐月下降。",s["cap"]),Spacer(1,4*mm),
           table([["月份","订单量（千单）","单位能耗（kWh/单）","能效状态"],["1月","120","8.4","基线"],["6月","176","7.5","改善中"],["9月","198","6.9","接近目标"],["12月","225","6.2","达标"]],[35*mm,48*mm,55*mm,36*mm],font_size=9.4),PageBreak()]
    # Page 6 two charts.
    st += [Paragraph("5. 异常分布与停机原因",s["h1"]),RLImage(str(assets["stacked"]),width=174*mm,height=90*mm),
           Paragraph("图 4　各作业区异常类型构成。",s["cap"]),Spacer(1,2*mm),RLImage(str(assets["donut"]),width=174*mm,height=90*mm),
           Paragraph("图 5　全年 400 小时停机的原因占比。",s["cap"]),PageBreak()]
    # Page 7 complex table.
    st += [Paragraph("6. 分区运营明细",s["h1"]),Paragraph("表 2 同时包含数量、比率、阈值状态与责任人，适合测试表头关联和行列定位。",s["body"]),Spacer(1,4*mm),
           table([["区域","功能","订单量(千单)","平均温度","异常数","准时率","负责人","风险级别"],
                  ["A区","收货","310","5.2°C","23","98.8%","孟川","低"],["B区","冷藏","420","2.6°C","27","99.1%","叶岚","中"],
                  ["C区","冷冻","385","-18.4°C","29","98.6%","沈溪","高"],["D区","分拣","460","16.8°C","13","99.4%","顾言","低"],
                  ["E区","包装","295","18.5°C","22","99.0%","唐薇","中"],["F区","发货","250","12.2°C","8","99.6%","陆远","低"]],
                 [18*mm,23*mm,25*mm,23*mm,18*mm,20*mm,20*mm,22*mm],font_size=8.1),Spacer(1,5*mm),
           Paragraph("表下注释：C 区风险级别为“高”，并不代表其全年平均温度超标；风险评级还综合考虑设备老化、货值和故障恢复时间。异常数最多的也是 C 区，共 29 起。",s["small"]),Spacer(1,6*mm),
           Paragraph("跨表计算提示",s["h2"]),Paragraph("六个区域订单量合计 2,120 千单。D 区比 F 区多 210 千单；F 区准时率最高。",s["body"]),PageBreak()]
    # Page 8 heatmap and floorplan.
    st += [Paragraph("7. 温控热力与空间布局",s["h1"]),RLImage(str(assets["heatmap"]),width=174*mm,height=86*mm),
           Paragraph("图 6　传感器热力图；E5 为最高温热点 -14.2°C，超过安全阈值。",s["cap"]),Spacer(1,2*mm),
           RLImage(str(assets["floor"]),width=174*mm,height=86*mm),Paragraph("图 7　平面示意图；夜间补货路线从 A/B 通道穿过，终点靠近 C/F 之间。",s["cap"]),PageBreak()]
    # Page 9 response flow and risk table.
    st += [Paragraph("8. 应急响应流程与风险矩阵",s["h1"]),RLImage(str(assets["flow"]),width=174*mm,height=102*mm),Paragraph("图 8　一级事件响应流程。",s["cap"]),Spacer(1,3*mm),
           table([["风险事件","概率","影响","风险分","应对措施","责任人"],["冷机停机","中","高","12","切换备用机组","叶岚"],["全场断电","低","极高","10","启动柴油发电机","陆远"],
                  ["网络中断","中","中","9","切换 5G 专网","顾言"],["温控漂移","高","中","15","校准传感器并复测","沈溪"]],[30*mm,18*mm,18*mm,18*mm,62*mm,25*mm],font_size=8.4),PageBreak()]
    # Page 10 conclusions.
    st += [Paragraph("9. 结论与行动计划",s["h1"]),
           Paragraph("总体结论：天枢冷链中心已完成规模增长与能效优化的双重目标，但 C 区温控和设备老化仍是首要风险。",s["body"]),Spacer(1,5*mm),
           table([["优先级","行动项","截止日期","验收标准","负责人"],["P0","更换 C 区除霜控制器","2026-01-10","连续30天无温控漂移","沈溪"],["P1","部署机械故障预测模型","2026-02-28","机械停机时长下降20%","顾言"],
                  ["P1","优化夜间补货路线","2026-03-15","平均路径缩短12%","孟川"],["P2","升级培训与演练平台","2026-04-30","覆盖率达到100%","唐薇"]],[20*mm,56*mm,32*mm,47*mm,23*mm],font_size=8.8),Spacer(1,6*mm),
           Paragraph("最终审批人为运营副总裁程越。批准日期为 2026 年 1 月 5 日。",s["body"]),Spacer(1,5*mm),
           Paragraph("附录：图表数据均为为测试而构造的合成数据，不对应任何真实企业或园区。",s["small"])]
    doc.build(st,onFirstPage=page_footer,onLaterPages=page_footer)


def write_qa():
    items=[
        ("R001","title",1,"封面标题","PDF 的完整标题是什么？","天枢冷链中心 2025 多模态运营分析报告",["天枢冷链中心","2025","多模态运营分析报告"]),
        ("R002","metadata",1,"封面表格","文档编号和报告负责人分别是什么？","NEBULA-PDF-RICH-2025-05；苏衡",["NEBULA-PDF-RICH-2025-05","苏衡"]),
        ("R003","body",2,"执行摘要","中心位于哪里、何时投入正式运营？","南京江宁区；2025 年 2 月 17 日",["南京江宁区","2025年2月17日"]),
        ("R004","table",2,"关键指标表","2025 年全年订单量和单位能耗分别是多少？","212 万单；6.2 kWh/单",["212","6.2"]),
        ("R005","body",2,"关键解释","4 月冷链合规率下降的原因和修复日期是什么？","C 区除霜控制器参数漂移；4 月 19 日",["C区","除霜控制器","参数漂移","4月19日"]),
        ("R006","image",3,"图1","图片中工程师的安全帽和反光背心分别是什么颜色？","白色安全帽，橙色反光背心",["白色","橙色"]),
        ("R007","image",3,"图1","图片中有几台绿色移动机器人和几个银色保温货箱？","2 台绿色移动机器人，3 个银色保温货箱",["2","3","绿色","银色"]),
        ("R008","image",3,"图1","红色灭火器位于画面哪侧，其上方是什么标志？","画面左侧；黄色三角警告标志",["左侧","黄色","三角","警告"]),
        ("R009","chart",4,"图2","哪一指标在 4 月低于 98% 目标线？数值是多少？","冷链合规率，96.8%",["冷链合规率","96.8%"]),
        ("R010","chart",4,"图2","设备可用率在哪个月达到峰值？峰值是多少？","11 月，99.6%",["11月","99.6%"]),
        ("R011","chart",4,"图2","12 月准时交付率是多少？","99.6%",["99.6%"]),
        ("R012","chart",5,"图3","订单量从 1 月到 12 月增加多少千单？","增加 105 千单",["105"]),
        ("R013","chart",5,"图3","单位能耗从 1 月到 12 月下降多少？下降比例约为多少？","下降 2.2 kWh/单，约 26.2%",["2.2","26.2%"]),
        ("R014","chart",5,"图3","折线图中哪个月首次达到低于 6.5 kWh/单的目标？","11 月",["11月"]),
        ("R015","chart",6,"图4","哪个区域异常总数最多？各类分别多少？","C 区最多；拣选错误14、包装破损6、温控偏差9，共29",["C区","14","6","9","29"]),
        ("R016","chart",6,"图4","A 区和 D 区的异常总数相差多少？","相差 10 起（A区23，D区13）",["10","23","13"]),
        ("R017","chart",6,"图5","停机原因占比最高的是什么？对应多少小时？","机械故障，140 小时",["机械故障","140"]),
        ("R018","chart",6,"图5","软件异常和网络中断合计占比多少？","37%",["37%"]),
        ("R019","table",7,"表2","C 区的功能、平均温度、异常数和负责人是什么？","冷冻；-18.4°C；29 起；沈溪",["冷冻","-18.4","29","沈溪"]),
        ("R020","table",7,"表2","哪个区域准时率最高？是多少？","F 区，99.6%",["F区","99.6%"]),
        ("R021","table_calculation",7,"表2","六个区域订单量合计多少？","2,120 千单",["2120"]),
        ("R022","heatmap",8,"图6","热力图中最高温热点位于哪个传感器？温度是多少？","E5，-14.2°C",["E5","-14.2"]),
        ("R023","heatmap",8,"图6","E5 是否满足 ≤-18.0°C 的安全阈值？偏离多少？","不满足；高 3.8°C",["不满足","3.8"]),
        ("R024","image",8,"图7","平面图中的东门集合点位于哪一侧、靠近哪个功能区？","右下侧，靠近 F 区发货区",["右下","F区","发货"]),
        ("R025","diagram",9,"图8","一级事件响应流程中 S2、S3、S4 分别是什么？","S2 5分钟内确认；S3 15分钟首轮处置；S4 30分钟升级",["5分钟","15分钟","30分钟"]),
        ("R026","table",9,"风险矩阵","风险分最高的事件是什么，责任人是谁？","温控漂移，风险分15，沈溪",["温控漂移","15","沈溪"]),
        ("R027","cross_page",6,"图5+第10页行动表","针对最大停机原因采取什么行动，目标是什么？","部署机械故障预测模型；机械停机时长下降20%",["机械故障预测模型","下降20%"]),
        ("R028","table",10,"行动计划表","P0 行动项、截止日期和负责人是什么？","更换 C 区除霜控制器；2026-01-10；沈溪",["更换C区除霜控制器","2026-01-10","沈溪"]),
        ("R029","body",10,"结论","报告认为首要风险是什么？","C 区温控和设备老化",["C区","温控","设备老化"]),
        ("R030","body",10,"审批信息","最终审批人和批准日期是什么？","程越；2026 年 1 月 5 日",["程越","2026年1月5日"]),
    ]
    with QA_PATH.open("w",encoding="utf-8") as f:
        for qid,mod,page,loc,q,a,kw in items:
            f.write(json.dumps({"id":qid,"source":PDF_PATH.name,"page":page,"modality":mod,"locator":loc,"question":q,"answer":a,"required_keywords":kw},ensure_ascii=False)+"\n")


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    assets={k:ASSETS/f"{k}.png" for k in ["scene","line","combo","stacked","donut","heatmap","floor","flow"]}
    make_scene(assets["scene"]); make_line_chart(assets["line"]); make_combo_chart(assets["combo"])
    make_stacked(assets["stacked"]); make_donut(assets["donut"]); make_heatmap(assets["heatmap"])
    make_floorplan(assets["floor"]); make_flow(assets["flow"])
    build_pdf(assets); write_qa()
    readme=(
        "# 丰富 PDF 多模态 RAG 样本\n\n"
        "- PDF：`05_天枢冷链中心多模态运营分析报告.pdf`（10 页）\n"
        "- 标准题：`05_天枢冷链中心_ground_truth.jsonl`（30 题）\n"
        "- 资产：`assets/`（现场图、多系列折线图、双轴组合图、堆叠图、环形图、热力图、平面图、流程图）\n\n"
        "索引时只输入 PDF，不要将标准题或 assets 目录加入知识库。\n"
    )
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    print(json.dumps({"pdf":PDF_PATH.name,"qa":QA_PATH.name,"questions":30},ensure_ascii=True))


if __name__ == "__main__":
    main()
