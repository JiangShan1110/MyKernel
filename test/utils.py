import logging, sys
logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s %(name)s: %(message)s",
                    stream=sys.stdout)   
LOG = logging.getLogger(__name__)

class Col:
    R = '\033[31m'  # 红
    G = '\033[32m'  # 绿
    B = '\033[34m'  # 蓝
    Y = '\033[33m'  # 黄
    M = '\033[35m'  # 洋红
    C = '\033[36m'  # 青
    RESET = '\033[0m'