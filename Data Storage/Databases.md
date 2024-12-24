# Databases

Working efficiently with databases requires knowing practical tricks, performance optimization tips, and best practices. Below are key database tips that will make your data operations faster, safer, and more maintainable.

---

## 1. **Performance Optimization**
- **Indexing**:
  - Always create indexes on columns used in `WHERE`, `JOIN`, and `ORDER BY` clauses.
  - Use **Composite Indexes** when queries filter based on multiple fields.
    ```sql
    CREATE INDEX idx_user_name_email ON Users (Name, Email);
    ```
  - Use **Covering Indexes** for queries that only reference indexed columns.

- **EXPLAIN Plans**:
  - Use `EXPLAIN` or `EXPLAIN ANALYZE` to debug slow queries and understand query execution plans.
    ```sql
    EXPLAIN SELECT * FROM Orders WHERE OrderDate > '2023-01-01';
    ```

- **Avoid SELECT***:
  - Retrieve only required columns to save memory and speed up execution.
    ```sql
    SELECT Name, Email FROM Users WHERE Status = 'Active';
    ```

- **Batch Updates/Inserts**:
  - Perform bulk operations rather than individual inserts or updates to avoid frequent I/O overhead.
    ```sql
    INSERT INTO Orders (OrderId, CustomerId) VALUES 
    (1, 101), (2, 102), (3, 103);
    ```

- **Partitioning**:
  - Use table partitioning (range, list, or hash) for very large datasets to optimize query performance.
    ```sql
    CREATE TABLE Sales (
        SaleId INT,
        SaleDate DATE
    )
    PARTITION BY RANGE (YEAR(SaleDate)) (
        PARTITION p2023 VALUES LESS THAN (2024),
        PARTITION p2024 VALUES LESS THAN MAXVALUE
    );
    ```

---

## 2. **SQL Tricks**
- **Find Duplicate Records**:
    ```sql
    SELECT Name, COUNT(*) 
    FROM Users 
    GROUP BY Name 
    HAVING COUNT(*) > 1;
    ```

- **Retrieve Top N for Each Group**:
  - Example: Get the latest orders for each customer.
    ```sql
    SELECT o.*
    FROM Orders o
    JOIN (
        SELECT CustomerId, MAX(OrderDate) as LatestOrder 
        FROM Orders 
        GROUP BY CustomerId
    ) recent ON o.CustomerId = recent.CustomerId AND o.OrderDate = recent.LatestOrder;
    ```

- **Dynamic Pivot Tables**:
    ```sql
    SELECT
        Product,
        SUM(CASE WHEN YEAR(SaleDate) = 2023 THEN Quantity ELSE 0 END) AS Sales2023,
        SUM(CASE WHEN YEAR(SaleDate) = 2024 THEN Quantity ELSE 0 END) AS Sales2024
    FROM Sales
    GROUP BY Product;
    ```

- **Recursive Queries** (e.g., Hierarchical or Tree-like Data):
    ```sql
    WITH RECURSIVE EmployeeHierarchy AS (
        SELECT EmployeeId, ManagerId, Name
        FROM Employees
        WHERE ManagerId IS NULL
        UNION ALL
        SELECT e.EmployeeId, e.ManagerId, e.Name
        FROM Employees e
        INNER JOIN EmployeeHierarchy eh
        ON e.ManagerId = eh.EmployeeId
    )
    SELECT * FROM EmployeeHierarchy;
    ```

---

## 3. **Database Design Tips**
- **Normalize for Stability**:
  - Use normalization to avoid redundancy and ensure scalability. Aim for **3NF** (Third Normal Form).

- **Denormalize for Performance**:
  - In highly transactional systems or read-heavy reporting, consider **denormalization** (e.g., summary tables or materialized views).
    ```sql
    CREATE MATERIALIZED VIEW SalesSummary AS
    SELECT StoreId, SUM(Sales) AS TotalSales
    FROM Sales
    GROUP BY StoreId;
    ```

- **Primary Keys**:
  - Always define primary keys for unambiguous row identification.
    ```sql
    CREATE TABLE Users (
        UserId INT PRIMARY KEY,
        Name VARCHAR(100)
    );
    ```

- **Foreign Keys with Constraints**:
  - Use foreign keys to maintain referential integrity.
    ```sql
    ALTER TABLE Orders ADD CONSTRAINT fk_customer FOREIGN KEY (CustomerId) REFERENCES Customers(CustomerId);
    ```

---

## 4. **Security Best Practices**
- **Parameterize Queries**:
  - Prevent SQL Injection by avoiding string concatenation.
    ```python
    # Python Example (Using Parameterized Queries)
    query = "SELECT * FROM Users WHERE Name = %s"
    cursor.execute(query, (user_input,))
    ```

- **Use Role-Based Access Control (RBAC)**:
  - Grant minimal privileges to users/groups based on their role.
    ```sql
    GRANT SELECT ON Employees TO ReportingRole;
    ```

- **Encrypt Sensitive Fields**:
  - For sensitive data, use in-database encryption or application-side encryption.
    ```sql
    CREATE TABLE Users (
        UserId INT,
        Email VARBINARY(256)  -- Encrypted field
    );
    ```

- **Audit Logs**:
  - Enable database logging for security audits.
    ```sql
    CREATE TABLE AuditLog (
        LogId INT PRIMARY KEY,
        EventType VARCHAR(100),
        EventTime TIMESTAMP
    );
    ```

- **Avoid Hardcoding Secrets**:
  - Store database credentials in environment variables or secret managers.

---

## 5. **Common Efficiency Tips**
- **Load Testing**:
  - Test read/write speeds using benchmarking tools like Apache JMeter or sysbench.

- **Caching**: 
  - Use in-memory data stores (e.g., Redis, Memcached) for frequently accessed, read-heavy queries.

- **ARCHIVE Storage**:
  - For infrequently used data, store rows in an **ARCHIVE table** to conserve resources.

- **Connection Pooling**:
  - Use libraries like **SQLAlchemy** or **HikariCP** for pooled connection management.

---

## 6. **NoSQL-Specific Tips**
- **MongoDB Query Optimization**:
  - Index fields used in `find()` or `aggregate()` pipelines.
    ```javascript
    db.users.createIndex({ "age": 1 })
    ```

- **Denormalization in NoSQL**:
  - Store embedded documents for performance:
    ```json
    {
        "customerId": 101,
        "name": "John Doe",
        "orders": [
            { "orderId": 201, "amount": 150 },
            { "orderId": 202, "amount": 200 }
        ]
    }
    ```

- **Cassandra**: Optimize partition keys and clustering columns to avoid performance bottlenecks in wide tables.

---

## 7. **Backup & Maintenance Tips**
- **Scheduled Backups**:
  - Automate daily database backups using tools (e.g., `pg_dump` for PostgreSQL).
    ```bash
    pg_dump -U postgres -F c -b -v -f backup_file.sqlc database_name
    ```

- **Vacuum (PostgreSQL)**:
  - Reclaim storage and prevent bloat.
    ```sql
    VACUUM (VERBOSE, ANALYZE);
    ```

- **Rebuild Indexes**:
  - Schedule periodic index rebuilds for large or frequently updated datasets.
    ```sql
    REINDEX TABLE my_table;
    ```

---

## 8. **Hidden Gems**
- **Window/Analytical Functions**:
  - Perform calculations across rows without grouping.
    ```sql
    SELECT Name, OrderDate, SUM(Amount) OVER (PARTITION BY CustomerId ORDER BY OrderDate) AS RunningTotal
    FROM Orders;
    ```

- **JSON Storage**:
  - Store and query JSON in relational databases (e.g., PostgreSQL).
    ```sql
    SELECT Name, Data->>'age' AS Age FROM Users;
    ```

- **Parallel Queries (PostgreSQL)**:
  - Enable parallel workers for large datasets:
    ```sql
    SET max_parallel_workers_per_gather = 4;
    ```

---

These tricks will help you gain better performance, scalability, and maintainable designs in your database workflows, while also ensuring security and reliability.