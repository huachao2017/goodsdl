DROP PROCEDURE IF EXISTS do_import_classify;
DELIMITER //
  CREATE PROCEDURE do_import_classify()
    BEGIN
		DECLARE v_first_code INT;
		DECLARE v_first_name VARCHAR(32);
		DECLARE v_second_code INT;
		DECLARE v_second_name VARCHAR(32);
		DECLARE v_id INT;
		DECLARE s int DEFAULT 0;
  
		DECLARE c_tmp CURSOR FOR SELECT a.first_code,a.first_name FROM goods_classify_excel as a GROUP BY first_code;
		DECLARE c2_tmp CURSOR FOR SELECT a.first_code,a.first_name,a.second_code,a.second_name,c.id FROM goods_classify_excel as a ,goods_classify as c where a.first_code = c.sort;

		DECLARE CONTINUE HANDLER FOR SQLSTATE '02000' SET s=1;
		open c_tmp;
			fetch c_tmp into v_first_code, v_first_name;
			while s <> 1 DO
				insert into goods_classify (corp_id,name,sort,create_time,modify_time,level,STATUS) VALUES (2118,v_first_name,v_first_code,NOW(),NOW(),0,10);
				fetch c_tmp into v_first_code, v_first_name;
			end while;
		close c_tmp;

		SET s=0;
		open c2_tmp;
			fetch c2_tmp into v_first_code, v_first_name,v_second_code, v_second_name,v_id;
			while s <> 1 DO
				insert into goods_classify (corp_id,name,sort,pid,create_time,modify_time,level,STATUS) VALUES (2118,v_second_name,v_second_code,v_id,NOW(),NOW(),1,10);
				fetch c2_tmp into v_first_code, v_first_name,v_second_code, v_second_name,v_id;
			end while;
		close c2_tmp;
    END;
    //
DELIMITER ;

DROP PROCEDURE IF EXISTS do_import_goods;
DELIMITER //
  CREATE PROCEDURE do_import_goods()
    BEGIN
		DECLARE v_id INT;
		DECLARE v_upc VARCHAR(32);
		DECLARE v_name VARCHAR(32);
		DECLARE v_specification VARCHAR(32);
		DECLARE v_unit VARCHAR(32);
		DECLARE v_mini_order_num INT;
		DECLARE v_is_return VARCHAR(32);
		DECLARE v_shipping_price FLOAT;
		DECLARE v_price FLOAT;
		DECLARE v_goods_id int;

		DECLARE s int DEFAULT 0;
		DECLARE v_sales_attr VARCHAR(32) DEFAULT '0';
  
		DECLARE c_tmp CURSOR FOR SELECT c.id,a.upc,a.name,a.specification,a.unit,a.mini_order_num,a.is_return,a.shipping_price,a.price FROM goods_excel as a, goods_classify as c where a.second_code = c.sort;

		DECLARE CONTINUE HANDLER FOR SQLSTATE '02000' SET s=1;
		open c_tmp;
			fetch c_tmp into v_id, v_upc, v_name, v_specification, v_unit, v_mini_order_num, v_is_return, v_shipping_price, v_price;
			while s <> 1 DO
				SET v_sales_attr = substring(v_is_return,1,1);
				if v_sales_attr = '0'
				THEN
						SET v_sales_attr = '0,78';
				ELSE
						SET v_sales_attr = '0';
				END IF;
				insert into goods (goods_upc,upc,name,status,price,distribution_price,create_time,modify_time,specification,corp_id,sales_attr,goods_classify_id) VALUES (v_upc,v_upc,v_name,10,v_price*100,v_shipping_price*100,NOW(),NOW(),CONCAT(v_specification,'/',v_unit),2118,v_sales_attr,v_id);
				SET v_goods_id=LAST_INSERT_ID();
				insert into goods_extend_erp (goods_id,basic_unit,packing_unit,packing_number,purchasing_unit,purchase_quantity,retail_unit,inventory_unit) VALUES (v_goods_id,0,0,0,0,v_mini_order_num,0,0);
				insert into shop_goods (shop_id,goods_id,name,price,cost_price,purchase_price,status,create_time,modify_time,goods_upc,upc,saleunit) values (4091,v_goods_id,v_name,v_price*100,v_price*100,v_shipping_price*100,10,NOW(),NOW(),v_upc,v_upc,v_unit);
				fetch c_tmp into v_id, v_upc, v_name, v_specification, v_unit, v_mini_order_num, v_is_return, v_shipping_price, v_price;
			end while;
		close c_tmp;

    END;
    //
DELIMITER ;

CALL do_import_classify();
CALL do_import_goods();