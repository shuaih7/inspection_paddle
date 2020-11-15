import paddlelite.lite as lite

a=lite.Opt()
# 非combined形式
a.set_model_dir(r"C:\Users\shuai\Documents\GitHub\inspection_paddle\develop\paddle-lite\mobilenet_v1")

# conmbined形式
# a.set_model_file("D:\\YOU_MODEL_PATH\\mobilenet_v1\\__model__")
# a.set_param_file("D:\\YOU_MODEL_PATH\\mobilenet_v1\\__params__")

a.set_optimize_out("mobilenet_v1_opt")
a.set_valid_places("x86")

a.run()