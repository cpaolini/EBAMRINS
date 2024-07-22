--CREATE TABLE patch_database (step INTEGER NOT NULL, X REAL NOT NULL,Y REAL NOT NULL, 
--Probability REAL NOT NULL, Width REAL NOT NULL, Height REAL NOT NULL);

ALTER TABLE patch_database
DROP COLUMN Width;



-- INSERT INTO patch_database (step, X, Y, Probability, Width, Height) 
-- VALUES (step, {A[origin,0]}, {A[origin,1]}, ZZ, xx, yy);

-- -- SELECT * FROM patch_database;