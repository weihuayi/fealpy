import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler
import pandas as pd

x = da.random.random((10000, 10000), chunks=(1000, 1000))

with Profiler() as prof, ResourceProfiler() as rprof:
    result = x.mean().compute()

# 打印 profiler 和 resource profiler 的结果
print(prof.results)
print(rprof.results)

# 将结果保存到 CSV 文件中
prof_df = pd.DataFrame(prof.results)
rprof_df = pd.DataFrame(rprof.results)

prof_df.to_csv("profiler_results.csv")
rprof_df.to_csv("resource_profiler_results.csv")

# 生成一个简单的 HTML 报告
with open("dask_report.html", "w") as f:
    f.write("<h1>Dask Profiler Report</h1>")
    f.write("<h2>Profiler Results</h2>")
    f.write(prof_df.to_html())
    f.write("<h2>Resource Profiler Results</h2>")
    f.write(rprof_df.to_html())

