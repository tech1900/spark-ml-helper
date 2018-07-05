# spark-sql
`Apply Range Paritioner to reduce data skew`
  
Test case demonstrate ways to change partitioner on spark sql.
_com.tech1900.spark.sql.RangePartitionStrategyTest_


```
*(2) HashAggregate(keys=[(id#0L % 10)#11L], functions=[count(1)], output=[key#2L, count(id)#9L])
 +- Exchange rangepartitioning((id#0L % 10)#11L ASC NULLS FIRST, 5)
    +- *(1) HashAggregate(keys=[(id#0L % 10) AS (id#0L % 10)#11L], functions=[partial_count(1)], output=[(id#0L % 10)#11L, count#13L])
       +- *(1) Range (0, 2147483647, step=1, splits=5)
```