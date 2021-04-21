## 1 数据库和SQL

### 1.3 SQL概要

#### SQL语句及种类

可以分为以下三类：

- DDL（Data Definition Language，数据定义语言）用来创建或者删除存储数据用的数据库以及表等对象

  CREATE、DROP、ALTER

- DML（Data Manipulation Language）用来查询或变更表中的记录

  SELECT、INSERT、UPDATE、DELETE

- DCL（Data Control Language，数据控制语言）用来确认或取消对数据库中的数据进行的变更。

  COMMIT、ROLLBACK、GRANT、REVOKE

#### SQL的基本书写规则

以分号结尾、不区分大小写、日期和字符串用单引号、单词用半角空格或换行分隔

### 1.4 创建表

#### 创建数据库

```SQL
CREATE DATABASE <数据库名称>

CREATE DATABASE shop;
```

#### 创建表

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

#### 数据类型

INTEGER 整型

NUMERIC 全体位数+小数位数

CHAR 定长字符型（不够长度空格补齐）

VARCHAR 变长字符型（不够长度不补空格）

DATE 日期型

#### 约束设置

NOT NULL非空约束

```SQL
product_id CHAR(4) NOT NULL,
```

主键约束

```SQL
PRIMARY KEY (product_id)
```

### 1.5 删除和更新表

#### 删除表

```SQL
DROP TABLE <表名>;

DROP TABLE Product;
```

DROP之后无法回撤。

#### 更新表的定义

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

## 2 查询基础

### 2.1 SELECT语句基础

#### 列的查询

```SQL
SELECT <列名>,……
  FROM <表名>;

SELECT product_id, product_name, purchase_price
  FROM Product;
```

查询结果中列的顺序和SELECT子句中的顺序相同。

#### 查询表中所有的列

```SQL
SELECT　*
FROM <表名>;
SELECT *
FROM Product;
```

#### 为列设定别名

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

#### 常数查询

```SQL
SELECT '商品' AS string, 38 AS number, '2009-02-24' AS date,
product_id, product_name
FROM Product;
```

#### 从结果删除重复行

使用DISTINCT

```SQL
SELECT DISTINCT product_type
FROM Product;
```

NULL也会被视为一类数据。另外，DISTINCT关键字只能用在第一个列名之前。

#### 根据WHERE语句选择记录

```SQL
SELECT <列名>, ……
FROM <表名>
WHERE <条件表达式>;

SELECT product_name, product_type
FROM Product
WHERE product_type = '衣服';
```

首先通过WHERE查询出符合指定条件的记录，再选取SELECT指定的列。另外，SQL中子句的书写顺序是固定的，也就是说WHERE子句必须紧跟FROM子句之后。

#### 注释

单行注释和多行注释：

```SQL
-- 本SELECT语句会从结果中删除重复行。
/* 本SELECT语句，
会从结果中删除重复行。*/
```

### 2.2 算术运算符和比较运算符

#### 算术运算符

```SQL
SELECT product_name, sale_price,
sale_price * 2 AS "sale_price_x2"
FROM Product;
```

#### 注意NULL

所有包含NULL的计算结果也是NULL。

#### 比较运算符

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

#### 对字符串使用不等号

chr字符串类型排序使用的是字典顺序。

#### 对NULL使用比较运算符

无论是使用=还是<>或者=NULL，均无法输出结果，要想选取NULL的记录应当使用IS NULL运算符。

```SQL
SELECT product_name, purchase_price
FROM Product
WHERE purchase_price IS NULL;

SELECT product_name, purchase_price
FROM Product
WHERE purchase_price IS NOT NULL;(这里直接省略IS NOT NULL似乎也可以)
```

### 2.3 逻辑运算符

#### NOT运算符

```SQL
SELECT product_name, product_type, sale_price
FROM Product
WHERE NOT sale_price >= 1000;
```

####  AND和OR运算符

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

#### 使用括号

```SQL
SELECT product_name, product_type, regist_date
FROM Product
WHERE product_type = '办公用品'
AND ( regist_date = '2009-09-11'
OR regist_date = '2009-09-20');
```

#### 含有NULL时的真值

存在不确定（UNKNOWN）的三值逻辑。

## 3 聚合与排序

### 3.1 对表进行聚合查询

#### 聚合函数

用于汇总的函数称聚合函数如COUNT、SUM、AVG、MAX、MIN。

#### 计算行数

```SQL
SELECT COUNT(*)
FROM Product;
```

1.count(1)与count(*)得到的结果一致，包含null值。
2.count(字段)不计算null值
3.count(null)结果恒为0

#### 计算NULL之外的数据的行数

```SQL
SELECT COUNT(purchase_price)
FROM Product;
```

*可以计算所有行数，而输入行名则只会得到NULL之外的行数，该特性是COUNT函数特有的，其他函数不能将\*作为参数。

#### 计算合计值

```SQL
SELECT SUM(sale_price)
FROM Product;
```

所有的聚合函数如果以列名为参数，那么在计算之前会把NULL排除在外。

#### 计算平均值

```SQL
SELECT AVG(sale_price)
FROM Product;
```

#### 计算最大值和最小值

```SQL
SELECT MAX(sale_price), MIN(purchase_price)
FROM Product;
```

另外，SUM和AVG函数只能对数值类型的列使用，但MAX和MIN函数可以适用于任何数据类型的列。

```SQL
SELECT MAX(regist_date), MIN(regist_date)
FROM Product;
```

#### 使用聚合函数删除重复值（DISTINCT）

```SQL
SELECT COUNT(DISTINCT product_type)
FROM Product;
```

对其他聚合函数也同理。

### 3.2 对表进行分组

#### GROUP BY子句

```SQL
SELECT <列名1>, <列名2>, <列名3>, ……
FROM <表名>
GROUP BY <列名1>, <列名2>, <列名3>, ……;

SELECT product_type, COUNT(*)
FROM Product
GROUP BY product_type;
```

GROUP BY指定的列成为聚合键或者分组列。

#### 聚合键中包含NULL的情况

```SQL
SELECT purchase_price, COUNT(*)
FROM Product
GROUP BY purchase_price;
```

#### 使用WHERE子句时GROUP BY的执行结果

```SQL
SELECT purchase_price, COUNT(*)
FROM Product
WHERE product_type = '衣服'
GROUP BY purchase_price;
```

#### 与聚合函数和GROUP BY子句有关的常见错误

①使用聚合函数时，SELECT子句中只能存在常数、聚合函数、GROUP BY指定的列名（聚合键）三个元素。

②GROUP BY子句中不能写列的别名，因为SQL先执行GROUP BY语句再执行 SELECT，也就是说在执行GROUP BY的时候DBMS还并不知道SELECT定义的别名。（但实际上MySQL中可以）。

③GROUP BY子句的结果顺序可能是随机的。

④只有SELECT和HAVING以及ORDER BY能够使用聚合函数，WHERE中不能使用。

### 3.3 为聚合结果指定条件

#### HAVING子句

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

#### HAVING子句的构成要素

和包含GROUP BY子句的SELECT子句一样，能够使用的要素也只有三种：常数、聚合函数和聚合键。 

聚合键所对应的条件在HAVING和WHERE中都可以使用，更适合写在WHERE子句中，处理速度更快。

### 3.4 对查询结果进行排序

#### ORDER BY子句

```SQL
SELECT <列名1>, <列名2>, <列名3>, ……
FROM <表名>
ORDER BY <排序基准列1>, <排序基准列2>, ……

SELECT product_id, product_name, sale_price, purchase_price
FROM Product
ORDER BY sale_price;
```

#### 指定升序或降序

```SQL
SELECT product_id, product_name, sale_price, purchase_price
FROM Product
ORDER BY sale_price DESC;
```

#### 指定多个排序键

```SQL
SELECT product_id, product_name, sale_price, purchase_price
FROM Product
ORDER BY sale_price, product_id;
```

#### NULL的顺序

MySQL中NULL排在最开始。

#### 在排序键中使用显示用的别名

GROUP BY执行顺序在SELECT之前，所以不能使用定义的别名。但ORDER BY可以

```SQL
SELECT product_id AS id, product_name, sale_price AS sp, purchase_price
FROM Product
ORDER BY sp, id;
```

#### ORDER BY子句中可以使用的列

ORDER BY可以使用表中不包含在SELECT子句中的列。

```SQL
SELECT product_name, sale_price, purchase_price
FROM Product
ORDER BY product_id;
```

也可以使用聚合函数。

```SQL
SELECT product_type, COUNT(*)
FROM Product
GROUP BY product_type
ORDER BY COUNT(*);
```

SELECT子句中也不必包含COUNT(*)。

## 4 数据更新

### 4.1 数据的插入（INSERT）

在INSERT之前，我们要首先创建一个表。

```SQL
CREATE TABLE ProductIns
(product_id CHAR(4) NOT NULL,
product_name VARCHAR(100) NOT NULL,
product_type VARCHAR(32) NOT NULL,
sale_price INTEGER DEFAULT 0,
purchase_price INTEGER ,
regist_date DATE ,
PRIMARY KEY (product_id));
```

#### INSERT语句的基本语法

```SQL
INSERT INTO <表名> (列1, 列2, 列3, ……) VALUES (值1, 值2, 值3, ……);

INSERT INTO ProductIns (product_id, product_name, product_type, sale_price, purchase_price, regist_date) VALUES ('0001', 'T恤衫', '衣服', 1000, 500, '2009-09-20');
```

#### 列清单的省略

对表进行全列INSERT时，可以省略列清单。

```SQL
INSERT INTO ProductIns VALUES ('0005', '高压锅', '厨房用具', 6800, 5000, '2009-01-15');
```

#### 插入NULL

```SQL
INSERT INTO ProductIns (product_id, product_name, product_type, sale_price, purchase_price, regist_date) VALUES ('0006', '叉子', '厨房用具', 500, NULL, '2009-09-20');
```

#### 多行INSERT

```SQL
INSERT INTO ProductIns VALUES ('0009', '打孔器', '办公用品', 500, 320, '2009-09-11'),
('0010', '运动T恤', '衣服', 4000, 2800, NULL),
('0011', '菜刀', '厨房用具', 3000, 2800, '2009-09-20');
```

多行一个出错所有数据都不会插入。

#### 插入默认值

创建表时可以使用DEFAULT约束设置默认值。

```SQL
CREATE TABLE ProductIns
(product_id CHAR(4) NOT NULL,
sale_price INTEGER DEFAULT 0,
PRIMARY KEY (product_id));
```

在插入时，可以通过显式插入：

```SQL
INSERT INTO ProductIns (product_id, product_name, product_type, sale_price, purchase_price, regist_date) VALUES ('0007', '擦菜板', '厨房用具', DEFAULT, 790, '2009-04-28');
```

或者直接省略：

```SQL
INSERT INTO ProductIns (product_id, product_name, product_type, purchase_price, regist_date) VALUES ('0007', '擦菜板', '厨房用具', 790, '2009-04-28');
```

如果省略了没有设定默认值的列，该列的值就会被设定为NULL。如果省略了设置NOT NULL同时没有DEFAULT的列，INSERT语句就会出错。

#### 从其他表中复制数据

创建表

```SQL
CREATE TABLE ProductCopy
(product_id CHAR(4) NOT NULL,
product_name VARCHAR(100) NOT NULL,
product_type VARCHAR(32) NOT NULL,
sale_price INTEGER ,
purchase_price INTEGER ,
regist_date DATE ,
PRIMARY KEY (product_id));
```

复制数据

```SQL
INSERT INTO ProductCopy (product_id, product_name, product_type, sale_price, purchase_price, regist_date)
SELECT product_id, product_name, product_type, sale_price, purchase_price, regist_date
FROM Product;
```

创建商品种类表

```SQL
CREATE TABLE ProductType
(product_type VARCHAR(32) NOT NULL,
sum_sale_price INTEGER ,
sum_purchase_price INTEGER ,
PRIMARY KEY (product_type));
```

复制合计表

```SQL
INSERT INTO ProductType (product_type, sum_sale_price, sum_purchase_price)
SELECT product_type, SUM(sale_price), SUM(purchase_price)
FROM Product
GROUP BY product_type;
```

### 4.2 数据的删除（DELETE）

#### DROP TABLE和DELETE语句

DROP TABLE会完全删除表，但DELETE会留下表，删除表中数据。

#### DELETE语句的基本用法

```SQL
DELETE FROM <表名>;

DELETE FROM Product;
```

#### 指定删除对象的DELETE语句（搜索型DELETE）

```SQL
DELETE FROM <表名>
WHERE <条件>;

DELETE FROM Product
WHERE sale_price >= 4000;
```

#### 删除和舍弃

MySQL中的TRUNCATE语句可以删除表中全部数据，处理速度要比DELETE更快。

```SQL
TRUNCATE Product;
```

### 4.3 数据的更新（UPDATE）

```SQL
UPDATE <表名>
SET <列名> = <表达式>;

UPDATE Product
SET regist_date = '2009-10-10';
```

UPDATE也会对原取值为NULL的数据行生效。

#### 指定条件的UPDATE语句

```SQL
UPDATE <表名>
SET <列名> = <表达式>
WHERE <条件>;

UPDATE Product
SET sale_price = sale_price * 10
WHERE product_type = '厨房用具';
```

#### 使用NULL进行更新

```SQL
UPDATE Product
SET regist_date = NULL
WHERE product_id = '0008';
```

只有未设置NOT NULL约束和主键约束的列才可以清空为NULL。

#### 多列更新

```SQL
UPDATE Product
SET sale_price = sale_price * 10,
purchase_price = purchase_price / 2
WHERE product_type = '厨房用具';

UPDATE Product
SET (sale_price, purchase_price) = (sale_price * 10, purchase_price / 2)
WHERE product_type = '厨房用具';
```

（方法一注意有逗号，方法二在MySQL中似乎不可行）

### 4.4 事务

```SQL
START TRANSACTION;
DML语句1;
DML语句2;
DML语句3;
.. .
事务结束语句（COMMIT或者ROLLBACK）;

START TRANSACTION;
-- 将运动T恤的销售单价降低1000日元
UPDATE Product
SET sale_price = sale_price - 1000
WHERE product_name = '运动T恤';
-- 将T恤衫的销售单价上浮1000日元
UPDATE Product
SET sale_price = sale_price + 1000
WHERE product_name = 'T恤衫';
COMMIT;
```

COMMIT是提交改动，并且无法进行恢复；ROLLBACK则回撤进行过的DML操作，回滚到事务开始之前的状态。

#### ACID特性

原子性（Atomicity）：事务结束时更新处理要么全部执行要么完全不执行。
一致性（Consistency）：事务中的处理要满足数据库的约束。
隔离性（Isolation）：不同事务之间互不干扰，在一个事务结束前，对其他事务不可见。
持久性（Durability）：事务结束后，DBMS能够保证该时间点的数据状态被保存。

## 5 复杂查询

### 5.1 视图

使用视图时不会讲数据保存到存储设备中。

#### 创建视图的方法

```SQL
CREATE VIEW 视图名称(<视图列名1>, <视图列名2>, ……)
AS
<SELECT语句>

CREATE VIEW ProductSum (product_type, cnt_product)
AS
SELECT product_type, COUNT(*)
FROM Product
GROUP BY product_type;
```

使用视图

```SQL
SELECT product_type, cnt_product
FROM ProductSum;
```

以视图为基础创建视图

```SQL
CREATE VIEW ProductSumJim (product_type, cnt_product)
AS
SELECT product_type, cnt_product
FROM ProductSum
WHERE product_type = '办公用品';

SELECT product_type, cnt_product
FROM ProductSumJim;
```

另外，对多数DBMS来说，多重视图会降低SQL的性能。视图的使用也有两个限制：

①定义视图时不要使用ORDER BY子句

因为视图和表一样，数据行是没有顺序的，所以不是所有DBMS都支持在定义视图时使用ORDER BY（其实MySQL可以）。

②对视图进行更新

更新视图时要求定义视图满足一定的条件，即更新视图时能够唯一的确定更新原表的方法。换句话说，视图在SQL中的存储和调用函数就是一个SELECT语句，所谓的更新视图实际上是在直接更新原表。

#### 删除视图

```SQL
DROP VIEW 视图名称(<视图列名1>, <视图列名2>, ……);

DROP VIEW ProductSum;
DROP VIEW ProductSum CASCADE;
```

在MySQL中，删除某视图后，所有基于它创建出来的视图也无法再使用。若要将这些视图也一并删除，可以使用CASCADE选项；而如果再次创建删除的视图，那么这些视图也可以重新恢复。

### 5.2 子查询

子查询就是将用来定义视图的SELECT语句直接用于FROM子句当中。

```SQL
SELECT product_type, cnt_product
FROM ( SELECT product_type, COUNT(*) AS cnt_product
	FROM Product
	GROUP BY product_type ) AS ProductSum;
```

也可以多层嵌套

```SQL
SELECT product_type, cnt_product
FROM (SELECT *
	FROM (SELECT product_type, COUNT(*) AS cnt_product
		FROM Product
		GROUP BY product_type) AS ProductSum
		WHERE cnt_product = 4) AS ProductSum2;
```

#### 子查询的名称

使用子查询时，需要使用AS关键字设定名称。

#### 标量子查询

标量子查询（scalar subquery）能且只能返回1行1列的结果。

```SQL
SELECT product_id, product＿name, sale_price
FROM Product
WHERE sale_price > AVG(sale_price);
```

如果我们想查询销售单价高于平均值的商品，使用上面的代码则会出错，因为聚合函数不能在WHERE中使用。

首先考虑能够计算平均值的查询

```SQL
SELECT AVG(sale_price)
FROM Product;
```

在WHERE中使用子查询

```SQL
SELECT product_id, product_name, sale_price
FROM Product
WHERE sale_price > (SELECT AVG(sale_price)
					FROM Product);
```

#### 标量子查询的书写位置

能够使用常数或者列名的地方，包括SELECT、GROUP BY、HAVING、ORDER BY等，都可以使用标量子查询。

```SQL
SELECT product_id,
	product_name,
	sale_price,
    (SELECT AVG(sale_price)
    FROM Product) AS avg_price
FROM Product;

SELECT product_type, AVG(sale_price)
FROM Product
GROUP BY product_type
HAVING AVG(sale_price) > (SELECT AVG(sale_price)
							FROM Product);
```

写在SELECT中可以新建一列每一行输出聚合函数的值，写在HAVING中可以将每组的聚合函数值与总体的聚合函数值作比较。

### 5.3 关联子查询

核心思想是将多行的查询转换成关联的标量子查询

```SQL
SELECT product_type, product_name, sale_price
FROM Product AS P1
WHERE sale_price > (SELECT AVG(sale_price)
					FROM Product AS P2
					WHERE P1.product_type = P2.product_type);
```

习题5-4中，要求输出每一类售价的平均值，不难得到要在SELECT中使用关联子查询，其中的条件仍为P1.xx =P2.xx，因为SQL先执行FROM，再执行SQL，所以即使P1定义在SELECT后面，但是因为它在FROM中所以先执行。

## 6 函数、谓词、CASE表达式

### 6.1 函数

#### 算术函数

ABS 绝对值
MOD（被除数，除数） 求余
ROUND（数，保留小数位数） 四舍五入

#### 字符串函数

CONCAT(str1, str2, ...)  拼接字符串
LENGTH(str) 字符串长度
CHAR_LENGTH(str) 字符串长度

> LENGTH()是按照字节来统计的，**CHAR_LENGTH**()是按照字符来统计的。例如：一个包含5个字符且每个字符占两个字节的字符串而言，LENGTH()返回长度10，**CHAR_LENGTH**()返回长度是5；如果对于单字节的字符，则两者返回结果相同。

LOWER(str) 小写转换
UPPER(str) 大写转换
REPLACE(str1, str2, str3) 将str1中的str2替换成str3
SUBSTRING(str1 FROM start FOR length) 从str1中的第start个字符开始截取长度为length的子串

#### 日期函数

CURRENT_DATE 当前日期
CURRENT_TIME 当前时间
CURRENT_TIMESTAMP 当前日期和时间
EXTRACT(element FROM date)

```MySQL
SELECT CURRENT_TIMESTAMP,
		EXTRACT(YEAR FROM CURRENT_TIMESTAMP) AS year,
		EXTRACT(MONTH FROM CURRENT_TIMESTAMP) AS month,
		EXTRACT(DAY FROM CURRENT_TIMESTAMP) AS day,
		EXTRACT(HOUR FROM CURRENT_TIMESTAMP) AS hour,
		EXTRACT(MINUTE FROM CURRENT_TIMESTAMP) AS minute,
		EXTRACT(SECOND FROM CURRENT_TIMESTAMP) AS second;
```

DATEDIFF(date1,date2) 计算date1在date2之后多少天
YEAR(), MONTH(),... 计算年份月份

#### 转换函数

转换在SQL中指数据类型的转换和值的转换

CAST(before AS type) 将before转成type类型

```MySQL
SELECT CAST('0001' AS SIGNED INTEGER) AS int_col;
SELECT CAST('2009-12-14' AS DATE) AS date_col;
```

COALESCE(data1, data2, ...) 返回参数中左侧开始第1个不是NULL 的值

```MySQL
SELECT COALESCE(str2, 'NULL')
FROM SampleStr;
```

例如使用上述代码，可以将数据表中的NULL转换成'NULL'

### 6.2 谓词

#### 字符串的部分一致查询（LIKE）

```MySQL
WHERE str LIKE 'aa%'
WHERE str LIKE '%aa%'
WHERE str LIKE '%aa'
WHERE str LIKE 'aa__'
```

#### 范围查询（BETWEEN）

BETWEEN会包含两个临界值（也就是<=和>=）

```MySQL
WHERE sale_price BETWEEN 100 AND 1000;
```

#### 判断是否为NULL（IS NULL、IS NOT NULL） 

```MySQL
WHERE purchase_price IS NULL;
WHERE purchase_price IS NOT NULL;
```

#### OR的简便用法（IN）

```MySQL
WHERE purchase_price IN (320, 500, 5000);
WHERE purchase_price NOT IN (320, 500, 5000);
```

注意，如果IN后使用了NULL，那么结果和不使用NULL一样（or a=NULL不是TRUE）
而如果NOT IN后使用了NULL，则返回空集（and a<>NULL不是true）

#### 使用子查询作为IN谓词的参数

```MySQL
SELECT product_name, sale_price
FROM Product
WHERE product_id IN (SELECT product_id
					FROM ShopProduct
					WHERE shop_id = '000C');
```

也就是将子查询得到的结果作为IN的参数

#### EXIST谓词

EXIST只有一个参数，并且通常都会使用关联子查询作为参数。

```MySQL
SELECT product_name, sale_price
FROM Product AS P
WHERE EXISTS (SELECT *
			FROM ShopProduct AS SP
			WHERE SP.shop_id = '000C'
			AND SP.product_id = P.product_id);
```

其中SELECT *替换成任何列都不改变结果，因为EXIST只会判断是否存在满足子查询中WHERE子句指定的条件的记录。

### 6.3 CASE表达式

```MySQL
CASE WHEN <求值表达式> THEN <表达式>
	WHEN <求值表达式> THEN <表达式>
	WHEN <求值表达式> THEN <表达式>
	...
	ELSE <表达式>
END

SELECT product_name,
	CASE WHEN product_type = '衣服'
		THEN 'A ：' | | product_type
		WHEN product_type = '办公用品'
		THEN 'B ：' | | product_type
		WHEN product_type = '厨房用具'
		THEN 'C ：' | | product_type
		ELSE NULL
	END AS abc_product_type
FROM Product;
```

不同类型的和汇总到不同列上

```MySQL
SELECT SUM(CASE WHEN product_type = '衣服'
			THEN sale_price ELSE 0 END) AS sum_price_clothes,
		SUM(CASE WHEN product_type = '厨房用具'
			THEN sale_price ELSE 0 END) AS sum_price_kitchen,
		SUM(CASE WHEN product_type = '办公用品'
			THEN sale_price ELSE 0 END) AS sum_price_office
FROM Product;
```

简单表达式

```MySQL
SELECT product_name,
	CASE product_type
		WHEN '衣服' THEN 'A ：' | | product_type
		WHEN '办公用品' THEN 'B ：' | | product_type
		WHEN '厨房用具' THEN 'C ：' | | product_type
		ELSE NULL
	END AS abc_product_type
FROM Product;
```

MySQL中可以使用IF，但是往往较为繁琐。

## 7 集合运算

### 7.1 表的加减法

#### 求并集（UNION）

```MySQL
SELECT product_id, product_name
FROM Product
UNION
SELECT product_id, product_name
FROM Product2;
```

注意：进行求并集的对象列数必须相同，且列的类型也必须一致，另外可以使用任何SELECT语句，但ORDER BY子句只能在最后使用一次。

保留重复行可以使用ALL选项

```MySQL
SELECT product_id, product_name
FROM Product
UNION ALL
SELECT product_id, product_name
FROM Product2;
```

MySQL中没有直接求交集和差集的操作

### 7.2 连接

#### 内连接（INNER JOIN）

```MySQL
SELECT SP.shop_id, SP.shop_name, SP.product_id, P.product_name, P.sale_price
FROM ShopProduct AS SP INNER JOIN Product AS P
ON SP.product_id = P.product_id;
```

ON后面指定的是两张表连接所使用的列（连接键），连接条件还可以使用其他谓词。

#### 外连接（OUTER JOIN）

```MySQL
SELECT SP.shop_id, SP.shop_name, SP.product_id, P.product_name, P.sale_price
FROM ShopProduct AS SP RIGHT OUTER JOIN Product AS P
ON SP.product_id = P.product_id;
```

使用LEFT或RIGHT指定主表

#### 3张表以上的连接

```SQL
SELECT SP.shop_id, SP.shop_name, SP.product_id, P.product_name, P.sale_price, IP.inventory_quantity
FROM ShopProduct AS SP INNER JOIN Product AS P
ON SP.product_id = P.product_id
	INNER JOIN InventoryProduct AS IP
	ON SP.product_id = IP.product_id
WHERE IP.inventory_id = 'P001';
```

相当于先内连接SP和IP，在对得到的内连接集和P进行内连接。

#### 交叉连接（CROSS JOIN）

```MySQL
SELECT SP.shop_id, SP.shop_name, SP.product_id, P.product_name
FROM ShopProduct AS SP CROSS JOIN Product AS P;
```

生成笛卡尔积

## 8 SQL高级处理

### 8.1 窗口函数

窗口函数也称OLAP（OnLine Analytical Processing）函数

```MySQL
<窗口函数> OVER ([PARTITION BY <列清单>]
				ORDER BY <排序用列清单>)
```

能够作为窗口函数使用的函数有聚合函数和RANK、DENSE_RANK、ROW_NUMBER等专用窗口函数。

```MySQL
SELECT product_name, product_type, sale_price,
RANK () OVER (PARTITION BY product_type
ORDER BY sale_price) AS ranking
FROM Product;
```

PARTITION BY可以设定RANK()的范围（类似GROUP BY，只不过不能汇总），ORDER BY则指定了对哪一列、使用何种顺序进行排序。

#### 专用窗口函数

RANK()、DENSE_RANK()、ROW_NUMBER()都可以计算排序位次，但对于并列名次，三者处理方式不同：
RANK：1、1、1、4
DENSE_RANK：1、1、1、2
ROW_NUMBER：1、2、3、4

专用窗口函数无需参数，所以通常括号中是空的

窗口函数只能在SELECT子句中使用

#### 使用聚合函数作为窗口函数

