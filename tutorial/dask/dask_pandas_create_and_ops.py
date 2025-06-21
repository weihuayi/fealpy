import dask.dataframe as dd
import pandas as pd

# 从 Pandas DataFrame 创建 Dask DataFrame
pdf = pd.DataFrame({'x': range(1000), 'y': range(1000)})
ddf = dd.from_pandas(pdf, npartitions=5)

mean_x = ddf['x'].mean().compute()
max_y = ddf['y'].max().compute()

print(f"Mean of x: {mean_x}, Max of y: {max_y}")

grouped = ddf.groupby('x').sum()
result = grouped.compute()

print(result)


