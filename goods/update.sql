delete from goods_goods where image_id < 5212223;
delete from goods_problemgoods where image_id < 5212223;
delete from goods_image where id < 5212223;
OPTIMIZE TABLE `goods_goods`;
OPTIMIZE TABLE `goods_problemgoods`;
OPTIMIZE TABLE `goods_image`;

delete from goods_goods where image_id < 7180000;
delete from goods_problemgoods where image_id < 7180000;
delete from goods_image where id < 7180000;
OPTIMIZE TABLE `goods_goods`;
OPTIMIZE TABLE `goods_problemgoods`;
OPTIMIZE TABLE `goods_image`;