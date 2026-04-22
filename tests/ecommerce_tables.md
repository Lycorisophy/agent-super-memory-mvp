# 电商数据库表设计文档

## users（用户表）
| 字段名 | 类型     | 注释         | 键     | 关联关系 |
|--------|----------|--------------|--------|----------|
| id     | INT      | 用户唯一标识 | PRI    |          |
| username | VARCHAR(50) | 用户名     |        |          |
| phone  | VARCHAR(20) | 手机号       |        |          |
| city   | VARCHAR(50) | 所在城市     |        |          |
| created_at | DATETIME | 注册时间     |        |          |

## categories（商品分类表）
| 字段名 | 类型     | 注释           | 键     | 关联关系      |
|--------|----------|----------------|--------|---------------|
| id     | INT      | 分类唯一标识   | PRI    |               |
| name   | VARCHAR(50) | 分类名称       |        |               |
| parent_id | INT   | 父分类ID       | MUL    | categories.id |
| sort_order | INT  | 排序权重       |        |               |

## products（商品表）
| 字段名 | 类型         | 注释         | 键     | 关联关系      |
|--------|--------------|--------------|--------|---------------|
| id     | INT          | 商品唯一标识 | PRI    |               |
| name   | VARCHAR(100) | 商品名称     |        |               |
| category_id | INT      | 所属分类ID   | MUL    | categories.id |
| price  | DECIMAL(10,2)| 售价         |        |               |
| stock  | INT          | 库存数量     |        |               |
| created_at | DATETIME  | 上架时间     |        |               |

## orders（订单表）
| 字段名 | 类型         | 注释         | 键     | 关联关系      |
|--------|--------------|--------------|--------|---------------|
| id     | INT          | 订单唯一标识 | PRI    |               |
| user_id | INT         | 下单用户ID   | MUL    | users.id      |
| order_no | VARCHAR(32) | 订单编号     | UNI    |               |
| total_amount | DECIMAL(10,2) | 订单总金额 |        |               |
| status | TINYINT      | 状态：1待付款 2已付款 3已发货 4已完成 5已取消 | | |
| created_at | DATETIME  | 下单时间     |        |               |

## order_items（订单明细表）
| 字段名 | 类型         | 注释         | 键     | 关联关系      |
|--------|--------------|--------------|--------|---------------|
| id     | INT          | 明细唯一标识 | PRI    |               |
| order_id | INT        | 所属订单ID   | MUL    | orders.id     |
| product_id | INT      | 商品ID       | MUL    | products.id   |
| quantity | INT        | 购买数量     |        |               |
| unit_price | DECIMAL(10,2) | 购买时单价 |        |               |
| total_price | DECIMAL(10,2) | 小计金额   |        |               |