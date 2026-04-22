-- =============================================================================
-- 结构化数据库智能体演示库（与 tests/schema_sample_*.md 字段对齐）
-- 导入示例（在 shell 中执行，按提示输入密码）：
--   mysql -uroot -p < docs/mysql_schema_agent_demo.sql
-- =============================================================================

CREATE DATABASE IF NOT EXISTS agent_schema_demo
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE agent_schema_demo;

-- 1. 用户表
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '用户唯一标识',
    username VARCHAR(50) NOT NULL COMMENT '用户名',
    phone VARCHAR(20) DEFAULT NULL COMMENT '手机号',
    city VARCHAR(50) DEFAULT NULL COMMENT '所在城市',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间'
) COMMENT '用户表';

-- 2. 商品分类表
CREATE TABLE categories (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '分类唯一标识',
    name VARCHAR(50) NOT NULL COMMENT '分类名称',
    parent_id INT DEFAULT NULL COMMENT '父分类ID，指向本表id',
    sort_order INT DEFAULT 0 COMMENT '排序权重'
) COMMENT '商品分类表';

-- 3. 商品表
CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '商品唯一标识',
    name VARCHAR(100) NOT NULL COMMENT '商品名称',
    category_id INT NOT NULL COMMENT '所属分类ID，关联categories.id',
    price DECIMAL(10,2) NOT NULL COMMENT '售价',
    stock INT DEFAULT 0 COMMENT '库存数量',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '上架时间'
) COMMENT '商品表';

-- 4. 订单表
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '订单唯一标识',
    user_id INT NOT NULL COMMENT '下单用户ID，关联users.id',
    order_no VARCHAR(32) NOT NULL UNIQUE COMMENT '订单编号',
    total_amount DECIMAL(10,2) NOT NULL COMMENT '订单总金额',
    status TINYINT DEFAULT 1 COMMENT '状态：1待付款 2已付款 3已发货 4已完成 5已取消',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '下单时间'
) COMMENT '订单表';

-- 5. 订单明细表
CREATE TABLE order_items (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '明细唯一标识',
    order_id INT NOT NULL COMMENT '所属订单ID，关联orders.id',
    product_id INT NOT NULL COMMENT '商品ID，关联products.id',
    quantity INT NOT NULL COMMENT '购买数量',
    unit_price DECIMAL(10,2) NOT NULL COMMENT '购买时单价',
    total_price DECIMAL(10,2) NOT NULL COMMENT '小计金额'
) COMMENT '订单明细表';


-- 插入用户
INSERT INTO users (username, phone, city, created_at) VALUES
('张三', '13800138001', '北京', '2025-01-15 10:00:00'),
('李四', '13900139002', '上海', '2025-02-20 14:30:00'),
('王五', '13700137003', '广州', '2025-03-10 09:15:00'),
('赵六', '13600136004', '深圳', '2025-04-01 16:20:00');

-- 插入分类
INSERT INTO categories (name, parent_id, sort_order) VALUES
('电子产品', NULL, 10),
('手机', 1, 20),
('电脑', 1, 30),
('服装', NULL, 40),
('男装', 4, 50),
('女装', 4, 60);

-- 插入商品
INSERT INTO products (name, category_id, price, stock, created_at) VALUES
('iPhone 15 Pro', 2, 8999.00, 50, '2025-03-01 08:00:00'),
('华为 Mate 60 Pro', 2, 6999.00, 30, '2025-03-05 10:00:00'),
('MacBook Pro 14', 3, 14999.00, 20, '2025-02-10 09:00:00'),
('T恤（男款）', 5, 99.00, 200, '2025-04-01 14:00:00'),
('连衣裙', 6, 299.00, 80, '2025-04-02 11:00:00');

-- 插入订单
INSERT INTO orders (user_id, order_no, total_amount, status, created_at) VALUES
(1, 'ORD20250301001', 8999.00, 4, '2025-03-01 15:30:00'),
(2, 'ORD20250315002', 15998.00, 2, '2025-03-15 11:20:00'),
(1, 'ORD20250401003', 398.00, 1, '2025-04-01 17:00:00'),
(3, 'ORD20250405004', 14999.00, 3, '2025-04-05 09:45:00');

-- 插入订单明细
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
(1, 1, 1, 8999.00, 8999.00),
(2, 1, 1, 8999.00, 8999.00),
(2, 3, 1, 14999.00, 14999.00),
(3, 4, 2, 99.00, 198.00),
(3, 5, 1, 299.00, 299.00),
(4, 3, 1, 14999.00, 14999.00);
