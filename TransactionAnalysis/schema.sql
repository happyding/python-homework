DROP TABLE IF EXISTS card_holder CASCADE;
DROP TABLE IF EXISTS merchant_category CASCADE;
DROP TABLE IF EXISTS credit_card CASCADE;
DROP TABLE IF EXISTS merchant CASCADE;
DROP TABLE IF EXISTS transaction CASCADE;

CREATE TABLE card_holder (
  holder_id INT NOT NULL PRIMARY KEY,
  holder_name VARCHAR(50) NOT NULL
);

CREATE TABLE merchant_category (
  merchant_category_id INT NOT NULL PRIMARY KEY,
  merchant_category VARCHAR(50) NOT NULL
);

CREATE TABLE credit_card (
  card_number varchar(20) NOT NULL PRIMARY KEY,
  card_holder_id INT NOT NULL,
  FOREIGN KEY (card_holder_id) REFERENCES card_holder(holder_id)
);

CREATE TABLE merchant (
  merchant_id INT NOT NULL PRIMARY KEY,
  merchant_name VARCHAR(50) NOT NULL,
  merchant_category_id INT NOT NULL,
  FOREIGN KEY (merchant_category_id) REFERENCES merchant_category(merchant_category_id)
);


CREATE TABLE transaction (
  transaction_id INT NOT NULL PRIMARY KEY,
  transaction_date timestamp NOT NULL,
  transaction_amount numeric NOT NULL,
  transaction_card_number varchar(20) NOT NULL,
  transaction_merchant_id INT NOT NULL,
  FOREIGN KEY (transaction_card_number) REFERENCES credit_card(card_number),
  FOREIGN KEY (transaction_merchant_id) REFERENCES merchant(merchant_id)
);