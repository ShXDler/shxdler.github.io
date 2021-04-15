# 1 SQL语句

### SQL语句及种类

可以分为以下三类：

- DDL（Data Definition Language，数据定义语言）用来创建或者删除存储数据用的数据库以及表等对象

  CREATE、DROP、ALTER

- DML（Data Manipulation Language）用来查询或变更表中的记录

  SELECT、INSERT、UPDATE、DELETE

- DCL（Data Control Language，数据控制语言）用来确认或取消对数据库中的数据进行的变更。

  COMMIT、ROLLBACK、GRANT、REVOKE

### SQL的基本书写规则

以分号结尾、不区分大小写、日期和字符串用单引号、单词用半角空格或换行分隔

## 创建表

### 创建数据库

```SQL
CREATE DATABASE <数据库名称>

CREATE DATABASE shop;
```

### 创建表

```SQL
CREATE TABLE <表名>
(<列名1> <数据类型> <该列所需约束>,
<列名2> <数据类型> <该列所需约束>,
<列名3> <数据类型> <该列所需约束>,
<列名4> <数据类型> <该列所需约束>,
...
<该表的约束1>, <该表的约束2>,……);

CREATE TABLE Product
(product_id CHAR(4) NOT NULL,
product_name VARCHAR(100) NOT NULL,
product_type VARCHAR(32) NOT NULL,
sale_price INTEGER ,
purchase_price INTEGER ,
regist_date DATE ,
PRIMARY KEY (product_id));
```

### 数据类型

INTEGER 整型

CHAR 定长字符型（不够长度空格补齐）

VARCHAR 变长字符型（不够长度不补空格）

DATE 日期型

### 约束设置

NOT NULL非空约束

```SQL
product_id CHAR(4) NOT NULL,
```

主键约束

```SQL
PRIMARY KEY (product_id)
```

## 删除和更新表

### 删除表

```SQL
DROP TABLE <表名>;

DROP TABLE Product;
```

DROP之后无法回撤

### 更新表的定义

添加列

```SQL
ALTER TABLE <表名> ADD COLUMN <列的定义>;

ALTER TABLE Product ADD COLUMN product_name_pinyin VARCHAR(100);
```

删除列

```SQL
ALTER TABLE <表名> DROP COLUMN <列名>;

ALTER TABLE Product DROP COLUMN product_name_pinyin;
```

插入数据

```SQL
START TRANSACTION;
INSERT INTO Product VALUES ('0001', 'T恤衫', '衣服', 1000, 500, '2009-09-20');
INSERT INTO Product VALUES ('0002', '打孔器', '办公用品', 500, 320, '2009-09-11');
INSERT INTO Product VALUES ('0003', '运动T恤', '衣服', 4000, 2800, NULL);
INSERT INTO Product VALUES ('0004', '菜刀', '厨房用具', 3000, 2800, '2009-09-20');
INSERT INTO Product VALUES ('0005', '高压锅', '厨房用具', 6800, 5000, '2009-01-15');
INSERT INTO Product VALUES ('0006', '叉子', '厨房用具', 500, NULL, '2009-09-20');
INSERT INTO Product VALUES ('0007', '擦菜板', '厨房用具', 880, 790, '2008-04-28');
INSERT INTO Product VALUES ('0008', '圆珠笔', '办公用品', 100, NULL,'2009-11-11');
COMMIT;
```

重命名表

```SQL
RENAME TABLE Poduct to Product;
```

# 2 查询基础

## 2-1 SELECT语句基础

### 列的查询

```SQL
SELECT <列名>,……
  FROM <表名>;

SELECT product_id, product_name, purchase_price
  FROM Product;
```

查询结果中列的顺序和SELECT子句中的顺序相同。

### 查询表中所有的列

```SQL
SELECT　*
FROM <表名>;
SELECT *
FROM Product;
```

### 为列设定别名

```SQL
SELECT product_id AS id,
product_name AS name,
purchase_price AS price
FROM Product;

SELECT product_id AS "商品编号",
product_name AS "商品名称",
purchase_price AS "进货单价"
FROM Product;
```

中文的别名要用双引号。

### 常数查询

```SQL
SELECT '商品' AS string, 38 AS number, '2009-02-24' AS date,
product_id, product_name
FROM Product;
```

### 从结果删除重复行

使用DISTINCT

```SQL
SELECT DISTINCT product_type
FROM Product;
```

NULL也会被视为一类数据。另外，DISTINCT关键字只能用在第一个列名之前。

### 根据WHERE语句选择记录

```SQL
SELECT <列名>, ……
FROM <表名>
WHERE <条件表达式>;

SELECT product_name, product_type
FROM Product
WHERE product_type = '衣服';
```

首先通过WHERE查询出符合指定条件的记录，再选取SELECT指定的列。另外，SQL中子句的书写顺序是固定的，也就是说WHERE子句必须紧跟FROM子句之后。

### 注释

单行注释和多行注释：

```SQL
-- 本SELECT语句会从结果中删除重复行。
/* 本SELECT语句，
会从结果中删除重复行。*/
```

## 2.2 算术运算符和比较运算符

### 算术运算符

```SQL
SELECT product_name, sale_price,
sale_price * 2 AS "sale_price_x2"
FROM Product;
```

### 注意NULL

所有包含NULL的计算结果也是NULL。

### 比较运算符

```SQL
SELECT product_name, product_type
FROM Product
WHERE sale_price = 500;

SELECT product_name, product_type, regist_date
FROM Product
WHERE regist_date < '2009-09-27';

SELECT product_name, sale_price, purchase_price
FROM Product
WHERE sale_price - purchase_price >= 500;
```

### 对字符串使用不等号

chr字符串类型排序使用的是字典顺序。

### 对NULL使用比较运算符

无论是使用=还是<>或者=NULL，均无法输出结果，要想选取NULL的记录应当使用IS NULL运算符。

```SQL
SELECT product_name, purchase_price
FROM Product
WHERE purchase_price IS NULL;

SELECT product_name, purchase_price
FROM Product
WHERE purchase_price IS NOT NULL;(这里直接省略IS NOT NULL似乎也可以)
```

## 2.3 逻辑运算符

### NOT运算符

```SQL
SELECT product_name, product_type, sale_price
FROM Product
WHERE NOT sale_price >= 1000;
```

###  AND和OR运算符

```SQL
SELECT product_name, purchase_price
FROM Product
WHERE product_type = '厨房用具'
AND sale_price >= 3000;

SELECT product_name, purchase_price
FROM Product
WHERE product_type = '厨房用具'
OR sale_price >= 3000;
```

### 使用括号

```SQL
SELECT product_name, product_type, regist_date
FROM Product
WHERE product_type = '办公用品'
AND ( regist_date = '2009-09-11'
OR regist_date = '2009-09-20');
```

### 含有NULL时的真值

存在不确定（UNKNOWN）的三值逻辑。

# 3 聚合与排序

## 3.1 对表进行聚合查询

### 聚合函数

用于汇总的函数称聚合函数如COUNT、SUM、AVG、MAX、MIN。

### 计算行数

```SQL
SELECT COUNT(*)
FROM Product;
```

### 计算NULL之外的数据的行数

```SQL
SELECT COUNT(purchase_price)
FROM Product;
```

*可以计算所有行数，而输入行名则只会得到NULL之外的行数，该特性是COUNT函数特有的，其他函数不能将\*作为参数。

### 计算合计值

```SQL
SELECT SUM(sale_price)
FROM Product;
```

所有的聚合函数如果以列名为参数，那么在计算之前会把NULL排除在外。

### 计算平均值

```SQL
SELECT AVG(sale_price)
FROM Product;
```

### 计算最大值和最小值

```SQL
SELECT MAX(sale_price), MIN(purchase_price)
FROM Product;
```

另外，SUM和AVG函数只能对数值类型的列使用，但MAX和MIN函数可以适用于任何数据类型的列。

```SQL
SELECT MAX(regist_date), MIN(regist_date)
FROM Product;
```

### 使用聚合函数删除重复值（DISTINCT）

```SQL
SELECT COUNT(DISTINCT product_type)
FROM Product;
```

对其他聚合函数也同理。

## 3.2 对表进行分组

### GROUP BY子句

```SQL
SELECT <列名1>, <列名2>, <列名3>, ……
FROM <表名>
GROUP BY <列名1>, <列名2>, <列名3>, ……;

SELECT product_type, COUNT(*)
FROM Product
GROUP BY product_type;
```

GROUP BY指定的列成为聚合键或者分组列。

### 聚合键中包含NULL的情况

```SQL
SELECT purchase_price, COUNT(*)
FROM Product
GROUP BY purchase_price;
```

### 使用WHERE子句时GROUP BY的执行结果

```SQL
SELECT purchase_price, COUNT(*)
FROM Product
WHERE product_type = '衣服'
GROUP BY purchase_price;
```

### 与聚合函数和GROUP BY子句有关的常见错误

①使用聚合函数时，SELECT子句中只能存在常数、聚合函数、GROUP BY指定的列名（聚合键）三个元素。

②GROUP BY子句中不能写列的别名，因为SQL先执行GROUP BY语句再执行 SELECT，也就是说在执行GROUP BY的时候DBMS还并不知道SELECT定义的别名。（但实际上MySQL中可以）。

③GROUP BY子句的结果顺序可能是随机的。

④只有SELECT和HAVING以及ORDER BY能够使用聚合函数，WHERE中不能使用。

## 3.3 为聚合结果指定条件

### HAVING子句

WHERE子句只能指定记录的条件，不能用来指定组的条件，要对集合指定条件可以使用HAVING子句。

```SQL
SELECT <列名1>, <列名2>, <列名3>, ……
FROM <表名>
GROUP BY <列名1>, <列名2>, <列名3>, ……
HAVING <分组结果对应的条件>

SELECT product_type, COUNT(*)
FROM Product
GROUP BY product_type
HAVING COUNT(*) = 2;
```

### HAVING子句的构成要素

和包含GROUP BY子句的SELECT子句一样，能够使用的要素也只有三种：常数、聚合函数和聚合键。 

另外，聚合键所对应的条件更适合写在WHERE子句中。