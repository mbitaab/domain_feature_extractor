import os

selenium_address = os.getenv("SELENIUM_ADDRESS","http://127.0.0.1:4445")
number_proc = int(os.getenv("NUMBER_PROC",1))
cache_mode = bool(os.getenv("CACHE_MODE",False))