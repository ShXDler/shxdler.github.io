### LENGTH()和CHAR_LENGTH()的区别

LENGTH()是按照字节来统计的，**CHAR_LENGTH**()是按照字符来统计的。例如：一个包含5个字符且每个字符占两个字节的字符串而言，LENGTH()返回长度10，**CHAR_LENGTH**()返回长度是5；如果对于单字节的字符，则两者返回结果相同。

### 判断日期在2020年6月的多种方法

```MySQL
LEFT(program_date,7) - '2020-06'
DATE_FORMAT(program_date,'%Y-%m') = '2020-06'
EXTRACT(YEAR_MONTH FROM program_date) = '202006'
program_date BETWEEN '2020-06-01' AND '2020-06-30'
DATEDIFF('2020-06-30', program_date) BETWEEN 0 AND 29
YEAR(program_date) = 2020 AND MONTH(program_date) = 06
program_date LIKE '2020-06%'
program_date REGEXP '^2020-06'
```

### 输出分组计数最多的

```MySQL
SELECT customer_number
FROM Orders
GROUP BY customer_number
ORDER BY COUNT(*) DESC
LIMIT 1;
```

### 关联子查询转换

关联子查询所需时间可能较长，可以使用IN语句

```MySQL
SELECT player_id, device_id
FROM Activity AS A1
WHERE (player_id, event_date) IN (
    SELECT player_id, MIN(event_date)
    FROM Activity
    GROUP BY player_id
);
```

### 找重复

找重复用GROUP BY和COUNT汇总，选不为1的行

### 找聚合函数的最大值方法

```MySQL
SELECT seller_id
FROM Sales
GROUP BY seller_id
HAVING SUM(price) >= ALL(
    SELECT SUM(price) AS s
    FROM Sales
    GROUP BY seller_id
);
```

第一种方法先建立子查询，再使用ALL函数分别比较

```MySQL
SELECT seller_id
FROM Sales
GROUP BY seller_id
HAVING SUM(price) = (
    SELECT SUM(price) AS s
    FROM Sales
    GROUP BY seller_id
    ORDER BY s DESC
    LIMIT 1
);
```

第二种建立子查询后进行排序输出第一个（最大的）作为标量子查询，这样就可以直接进行比较了

第二种只需要进行一次排序，后面对每一行只需要一次比较，第一种则每次都要比较所有行，占用时间可能较多。

### 统计只在一定时间范围内出售过的商品

```MySQL
SELECT P.product_id, product_name
FROM Sales AS S LEFT JOIN Product AS P
    ON S.product_id = P.product_id
GROUP BY P.product_id
HAVING SUM(sale_date < '2019-01-01') = 0 AND SUM(sale_date > '2019-03-31') = 0;
```

### 统计符合某一条件的行

可以使用SUM(condition)简化程序