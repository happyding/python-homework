select cc.card_holder_id, 
date(tc.transaction_date), 
date_part('month', tc.transaction_date) as month, 
date_part('hour', tc.transaction_date) as hour, 
tc.transaction_amount, 
tc.transaction_merchant_id
from transaction tc, credit_card cc
where tc.transaction_card_number = cc.card_number and 
        ( cc.card_holder_id = 2 OR cc.card_holder_id = 18 OR cc.card_holder_id = 25 ) and
        ( date_part('hour', tc.transaction_date) >=7 ) and 
        ( date_part('hour', tc.transaction_date) <=9 ) 