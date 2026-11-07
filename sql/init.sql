-- -*- coding: utf-8 -*-
-- 学生库初始化脚本：建库 + 建表 + 12 行种子数据。
--
-- 两种用法（任选其一）：
--   1) Docker：docker compose up -d   （容器首次启动时自动执行本文件）
--   2) 本地 MySQL：mysql -u root -p < sql/init.sql
--
-- 幂等：重复执行不会报错，也不会覆盖已存在的行（INSERT IGNORE + 固定主键）。
-- 字段定义必须与 README 里描述的 schema 保持一致——
-- eval/eval_run.py 用 CREATE TABLE ... LIKE student_scores 克隆评测表，结构变了评测就偏。

CREATE DATABASE IF NOT EXISTS `school_db`
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE `school_db`;

CREATE TABLE IF NOT EXISTS `student_scores` (
  `id`           INT           NOT NULL AUTO_INCREMENT,
  `class_name`   VARCHAR(50)   NOT NULL COMMENT '班级',
  `student_no`   VARCHAR(30)   NOT NULL COMMENT '学号',
  `student_name` VARCHAR(20)   NOT NULL COMMENT '姓名',
  `chinese`      DECIMAL(5,2)  NOT NULL DEFAULT 0.00 COMMENT '语文 0~150',
  `math`         DECIMAL(5,2)  NOT NULL DEFAULT 0.00 COMMENT '数学 0~150',
  `english`      DECIMAL(5,2)  NOT NULL DEFAULT 0.00 COMMENT '英语 0~150',
  `created_at`   DATETIME      DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  -- 同班同学号不可重复：tools.add_student() 里也有一次友好校验，这里是最后一道防线
  UNIQUE KEY `uk_class_no` (`class_name`, `student_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 种子数据：3 个班 12 人。id 固定，README 里的示例「张三（id=1）」才能对上。
-- 与 eval/eval_tasks.py 的 SEED_ROWS 同源，改这里请同步改那边。
INSERT IGNORE INTO `student_scores`
  (`id`, `class_name`, `student_no`, `student_name`, `chinese`, `math`, `english`)
VALUES
  (1,  '高一(1)班', '2024001', '张三', 60.00, 99.00, 88.50),
  (2,  '高一(1)班', '2024002', '李四', 85.00, 72.00, 66.00),
  (3,  '高一(1)班', '2024003', '王五', 85.00, 92.00, 93.00),
  (4,  '高一(1)班', '2024004', '赵六', 78.00, 85.00, 80.00),
  (5,  '高一(1)班', '2024005', '钱七', 95.00, 90.00, 88.00),
  (6,  '高一(1)班', '2024006', '孙八', 80.00, 75.00, 82.00),
  (7,  '高一(2)班', '2024101', '陈一', 70.00, 66.00, 91.00),
  (8,  '高一(2)班', '2024102', '陈二', 88.00, 94.00, 79.00),
  (9,  '高一(2)班', '2024103', '陈三', 55.00, 60.00, 58.00),
  (10, '高二(3)班', '2025201', '刘备', 77.00, 83.00, 90.00),
  (11, '高二(3)班', '2025202', '关羽', 92.00, 88.00, 85.00),
  (12, '高二(3)班', '2025203', '张飞', 65.00, 70.00, 73.00);
