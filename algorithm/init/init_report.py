from report.Basic_F1 import report_Basic_F1
from report.Som_report import report_Som

report_function_dic = {
    "report_Basic_F1": report_Basic_F1,
    'report_Som': report_Som
}

def init_report_function(config, *args, **params):
    name = config.get("report", "report_fun")

    if name in report_function_dic:
        return report_function_dic[name]
    else:
        raise NotImplementedError
