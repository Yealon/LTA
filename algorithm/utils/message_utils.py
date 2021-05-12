
def gen_time_str(t):
    """
    转化为 '分钟 : 秒' 的格式
    :param t:    timer
    :return:    '%2d:%02d'
    """
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)

def correct_msg(msg):
    return f'     ✔ {msg}'
def warning_msg(msg):
    return f'     ︎⚠ {msg}'
def infor_msg(msg):
    return f'☛ {msg}'
def erro_msg(msg):
    return f'     ✘ {msg}'
def report_msg(msg):
    return f'     ⚑ {msg}'
def epoch_msg(**kwargs):
    #     print("Epoch  Stage  Iterations  Time/Usage    Loss    Output Information")
    output = '🎼'
    for k in kwargs:
        output += ' ' + str(k) + ': ' + str(kwargs[k])
    output += ' :||'
    return output

