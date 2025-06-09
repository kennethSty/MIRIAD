#!/bin/bash

# create directory
mkdir 20200705v1
mkdir 20200705v1/full/
mkdir 20200705v1/full/metadata/
mkdir 20200705v1/full/pdf_parses/

wget -O 20200705v1/LICENSE 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/LICENSE?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=8%2BdABXguI9RUaxvsuFM9g9%2B9Zk0%3D&Expires=1657934019'

wget -O 20200705v1/RELEASE_NOTES 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/RELEASE_NOTES?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Q0iT802vPs7N%2F1RdHuF6gBYSimo%3D&Expires=1657934019'

wget -O 20200705v1/full/metadata/metadata_0.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_0.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=fV6h9Cze306EG%2BXXSrJ7ctePFK4%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_1.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_1.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=eq8ozIUK5AjMKPFcsTDrA%2FeG8yw%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_10.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_10.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ReTmoF5FOTletKsktxLcVm3Q8Do%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_11.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_11.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ytXpDAA6WJXvpJwlF0F1gSkZEFY%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_12.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_12.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=nhThFucbvbXO30vJGEs0OxYphxE%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_13.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_13.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=LKxJB5wenIRGVs%2BKTJlRO%2Fmo6Fk%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_14.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_14.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=oMaqI0piI8QoIogYHrr5e3vS98o%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_15.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_15.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Rcfi%2BwCYrtY0tBIcNjCVr7DE4dk%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_16.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_16.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=0hobmLKttZ1EbRoYWthG2USFUOs%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_17.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_17.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=WAeiV6ucRACWN%2BAjXStbpIPU3V8%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_18.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_18.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=VY9mWBnj9gAutNWB4VqAYjrYAFs%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_19.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_19.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=LQZzZAlRcGjXHfHZMPyTwwPFILg%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_2.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_2.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=WzUXMK5PVVxXlGgljaR3SR1y8uA%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_20.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_20.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=owICNngsJdljkkZqZerbHgrYZDM%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_21.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_21.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=VpclzJPVYKdLNaPqy%2BDgb9GX4tA%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_22.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_22.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=nuxX1%2FIQfGTg%2Fqtj4NrnVER958A%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_23.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_23.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cJKEn%2BlxM8vB2QLlIuiq3ki5dGM%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_24.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_24.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=YXRoebsbNzpBXGKfIPiQvHXww44%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_25.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_25.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=zB5RCImobjNM7yOzNIcajplY2Gk%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_26.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_26.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=700TfahffOPaMix1lukrGzytfvw%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_27.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_27.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=WSOuQKoJhBP329Pr3Boza9JpmmU%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_28.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_28.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=VZXo7w%2BK%2BY14LGGl7O318Z7fza8%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_29.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_29.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=1GduYx5gDBEiYInTN9ONXSIfr8c%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_3.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_3.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=TZV27QaAcPa%2F2%2B1z1a%2Fmr7tnsas%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_30.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_30.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6bUVQyyYzHXDtuneswzKXKB%2FjUk%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_31.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_31.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=omHmZGGEAXaBuWXyn%2F%2BmwwlCIVw%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_32.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_32.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rbWDmdT%2Fm%2BQsokwIURSM7VO3uW4%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_33.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_33.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=NM%2BvIAy5jZTQEbISQFK3%2F6JZLXo%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_34.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_34.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=lYpIgQdGBLqPNenNKdd%2Bz1bYH9w%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_35.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_35.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=y0G9FftwRiwaCAzEXpNnGLbKH4w%3D&Expires=1657934020'

wget -O 20200705v1/full/metadata/metadata_36.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_36.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=iYvf%2FpBmbw0FEydMMhoLTc4bjG8%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_37.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_37.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6XePptnmdaWwV5NqfOi1VoYkxII%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_38.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_38.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=NpYm11Fus%2BkG17Rwm0CZDeiIzko%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_39.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_39.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=FJbAcWRoZaEFUWucUWxdPWEW5fs%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_4.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_4.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cmpwHxhfoi4VTr1bYC1msnGFw%2BU%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_40.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_40.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=HIX69ZwY%2FHjunR%2Br3wJgsIXniyM%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_41.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_41.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=sH%2FwFvH60iUEznUboJYyHwYbUss%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_42.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_42.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=61Vzu58GG3%2F%2BIALjrd8YDvEyJmU%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_43.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_43.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=gceWRoFXhxcScMc5XeDaa%2Fub%2BIo%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_44.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_44.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Xu3f36il3htQHofHmVpw%2Bd96zmU%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_45.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_45.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=fnCPtw30SqczbxQe0B%2FtM0hs9Ic%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_46.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_46.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=N63CnQP9qH2XhPjHLJKD5BGk0go%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_47.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_47.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=qU%2FSwKN1%2BGGm6FYAez1Lqj%2B7Slg%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_48.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_48.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=eb9Z2Z5Fxm78TqSjUt6bf0Ql1Eo%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_49.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_49.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=%2ByS6BxibkkCWIrXVbz%2BWYUAEmK0%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_5.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_5.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=UeZE56K8RAbnlB3buxjkGAl%2BJPU%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_50.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_50.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=pLEaQMNgY46pTw7yo7%2FKD8cXK6s%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_51.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_51.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=JdfSY%2B8SmP8oYsLYx6ZSLsQsnaU%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_52.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_52.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=v2XuBey8uh%2F8ZBc7Unsi%2FFioxyU%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_53.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_53.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=kATdO2HGHcFPhVGoUfJ8CnsvmJI%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_54.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_54.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=04VZBYz4kT9XwseJ4FK396ClzBg%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_55.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_55.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=PiHfOJO2zYcO8PwAXfRnulNnFWk%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_56.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_56.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=S8nZrII2heKTw0KE%2FFyvzBdg3%2FQ%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_57.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_57.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=0JvlWQRwvy56Yn6gH6W99P%2FhOXk%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_58.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_58.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=niTfkCD%2Fmlrd9GYu%2FXHlVxCOoUA%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_59.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_59.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=YoMU1w1RamyxR7OjVg9S4N0l4Fw%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_6.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_6.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Q5AXuwhN0t9ggNb0eNwgPpODm24%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_60.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_60.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=2R8astLYjjqYx%2BUT5AVux%2FfLzh4%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_61.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_61.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rTGzgbq4U%2B%2Bd5G3v2FRXFybztJQ%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_62.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_62.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=FzYTzZ9j5y2iniH4lSbT8BVfXc0%3D&Expires=1657934021'

wget -O 20200705v1/full/metadata/metadata_63.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_63.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=SbpYVQu0Qcaq7WraG%2F4%2FhKMJ8w0%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_64.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_64.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=GKJXik4mxIOm7INb6dt1Lc2%2B%2F28%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_65.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_65.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ud4%2BBfemwNvc%2BqHfY%2BoJlY%2Baiho%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_66.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_66.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=AULmd%2BAIMFFnNe4UlypJTh2aspA%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_67.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_67.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=eMMSI9RZL%2B99kjWDg%2Bparm2EjvI%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_68.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_68.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MVlQUI54Okie6Dsww7sbe9LCFSU%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_69.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_69.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=fY7v8EZzyNy0Ms1cRBE2qn0kcWs%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_7.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_7.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=PrHbwB4dfjuuklu72KbePh3971M%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_70.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_70.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=HLYlCICIV%2F8dqmmR2jiIODl0OdY%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_71.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_71.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=U1mNic9WWsMdHbcvOo%2B08n5H0rw%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_72.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_72.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=8Fhnswef4nM9FEsUokkyfdhVe7k%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_73.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_73.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=NyQaqKqF%2BDOIugk9I3JtB2jeRys%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_74.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_74.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=01cB4MrP5lr%2FQRvmrvyoG1c8kHA%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_75.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_75.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Xgk0m97jQ4O0DCx8pyrfxnvTtvE%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_76.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_76.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=bE6uEFrp3bG6KcpBm8%2B0LJvtAyc%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_77.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_77.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=dv8OOCocQLGdBffNkw6H9yYpWWc%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_78.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_78.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=DV4jTg7FFPQD5k52n79VWOYNOTc%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_79.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_79.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=xHCa1R7Cnt6F4OCBR%2Bk8ybNJdFY%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_8.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_8.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Du8H5oluBu4Ztvn%2FBBPzH7dWvSI%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_80.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_80.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=eVTO64nheBXvAUC81W0gOqL4ySY%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_81.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_81.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=IyNv2rKTN%2FsLekWx5OHVQgueIi4%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_82.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_82.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=iyu%2F1lTm028ygBz3o%2Bf9UI2%2For8%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_83.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_83.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=qrC0xjgPjj8QWpHi1aGV5wc%2BIoE%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_84.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_84.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=A1K4wIflwWElQLaUawt3%2FSubc3c%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_85.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_85.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=0%2BJ82kW6REqEixRaOnHzptJF%2BKA%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_86.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_86.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6K%2Bl1KLHbHPUhYD0YlLGeXEHhqE%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_87.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_87.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=rbr9pIfT%2Ba3T0nBDfiQ4mwUSHjg%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_88.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_88.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=udx%2FmgVAylp7FAdAZt97QMKzEAI%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_89.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_89.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=LOh9aMbdJNhJSZDzG3ALUffhees%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_9.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_9.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=97jRjN5jsmlr5c8Im2aFTwYkzkI%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_90.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_90.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=pZ9RI4OI7qUI%2B7CEq2CP4zGReP0%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_91.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_91.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=71DkqaulhuCnB9eB9aCrpR6fPQU%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_92.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_92.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=zVYr%2FZvX5fcsbH8FHxwic8D33ls%3D&Expires=1657934022'

wget -O 20200705v1/full/metadata/metadata_93.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_93.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=NtEifqGyXFdfvRYxAktvifJm8nc%3D&Expires=1657934023'

wget -O 20200705v1/full/metadata/metadata_94.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_94.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=BXvgm4ms%2FXmzuRxo5m9mHd%2FgF0I%3D&Expires=1657934023'

wget -O 20200705v1/full/metadata/metadata_95.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_95.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=yj594X%2FLdDzqQcCtVCwn1bqMpOE%3D&Expires=1657934023'

wget -O 20200705v1/full/metadata/metadata_96.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_96.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=7PQrIkeL0TelLHMmA9PupdLPZOk%3D&Expires=1657934023'

wget -O 20200705v1/full/metadata/metadata_97.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_97.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=fdYf9XNL3bgLTpN1rAniOIdkqsg%3D&Expires=1657934023'

wget -O 20200705v1/full/metadata/metadata_98.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_98.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=zaxpCBycuAQdrK6A2QBcL3AfzV0%3D&Expires=1657934023'

wget -O 20200705v1/full/metadata/metadata_99.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_99.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ZoOLvz659KpegPlnVxDSORrbZec%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_0.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_0.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=phSGg4QMGMG%2FllxHJixO5EAfG68%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_1.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_1.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=9qc9dmjkJdOytvSFgTV5%2FHmSE8o%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_10.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_10.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=9acjcZ4l0iQMdea1e3bRNVIHh1I%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_11.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_11.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=PkUURkS56knGBHv%2F0mLvoXe%2Bo4s%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_12.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_12.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Nc4NCGF5qrBbh0saU886FWnhIUE%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_13.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_13.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=%2BrMs5He2FRYok%2FLm3EVMCZlS1sI%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_14.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_14.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vievtJlXWEy7mucpA%2Bt%2BaBvhwMo%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_15.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_15.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=sMUk3VKdByMszzqM8rFlw5jsQrM%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_16.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_16.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=JMZAJPxgOMTQH5O1gTdzfuXj8nk%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_17.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_17.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Yqx8kr5GKpTid7nHFLUb47TejYo%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_18.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_18.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=OVVQrKrQ8Du3%2B04hrZJgixtNO9o%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_19.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_19.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=lYexG8SHwIJ4KSdrnFTQq%2BMywzA%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_2.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_2.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=k5%2BspGo1iinheEB3hMMUjQx2GZg%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_20.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_20.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=1ws5of1Ik99lzCERfphCYsAI%2BOw%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_21.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_21.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=1NvELcVwveG7frVOgCybdwwrP6s%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_22.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_22.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=buuUFLAQuleK%2BPBrYah1x0sCX88%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_23.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_23.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=nWKjGTiI6fHLyUqefYrgcTL1%2B9M%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_24.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_24.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=N90agCxg6IP%2BpSBtuFItfGbwbGI%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_25.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_25.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=9ETVcqps58BDfEbs2H9l4BKI024%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_26.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_26.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=f4kZvhpHiF5BnTTzgFhEfCwFVqU%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_27.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_27.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=KsQKZty6X9grzKUuDIOOjAz5RPw%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_28.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_28.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=BCaeowa1p3kDSnGl0UVO9FcHgG8%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_29.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_29.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=MVCcFrtxNl6YfZUVfrweDO9V5Ko%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_3.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_3.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=V44C3f%2FhSKA8dqojwAzAVtRaR6Q%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_30.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_30.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ogISnQucEr%2FvzlMbum2pBYwwgZI%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_31.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_31.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=yYY8up9mHUjF8JBOYqtJb4DnRMU%3D&Expires=1657934023'

wget -O 20200705v1/full/pdf_parses/pdf_parses_32.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_32.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=k1PtItSu86M9ZZiobmocb1wLH5Y%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_33.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_33.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=lRtR0H1L4mk884KmiKQeCkcwrYE%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_34.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_34.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=qA8n5grDQ4n%2BNrFgVcSpuzSRpXk%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_35.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_35.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=875TbmenadTZB7Tio%2BSPxjM5hN0%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_36.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_36.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=U73SlqQC0Q5MFJ3LKVjm9GcwXGo%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_37.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_37.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Gykj%2FhjZgdWxMq1Xb3vqwM5Yaf0%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_38.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_38.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=aZypLI4QCwPyd1GyRQgVFU1TvOg%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_39.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_39.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=NNPqGIOHuE4tX4hS%2BUtux%2BDWllI%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_4.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_4.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=LL5go5iEIpC%2Bj%2B5bHIlG7pwy2qw%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_40.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_40.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=8V%2BnhpQ%2Fxk0AVWIw4B18veSFGBA%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_41.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_41.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=hBYsy1UFZL7uMiVNHo2tV4an3cc%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_42.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_42.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=IVN%2B%2Bfmw40yutlDztUs64AMKX4U%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_43.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_43.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=3H4azpmpmY%2BWSStNN%2BQnBed2EGo%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_44.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_44.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=OgOiShSmujJ1YKYHYS5JmKL7Wps%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_45.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_45.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=qad2NGIuPlbHYgZeKGAH7OcDiLM%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_46.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_46.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=q2FbJ9kPk%2F0Y%2Bs4LPdj%2FEiImm0w%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_47.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_47.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=j87JRuWPxj4cFIBZf0xtk9HDHGE%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_48.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_48.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=mG%2F%2BgEW6jlSQtBmpSyAkhU2ta2s%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_49.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_49.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=kkXcooXMP248x8ehjaUUqBvv8Eg%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_5.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_5.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=8T0WhBSuScWeJJVrz3oSL%2BRZGwc%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_50.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_50.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Roe8uzQotDCD92KR7EhnXJ9QQww%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_51.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_51.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=GhZd4Pi9tLzN4%2BJ1bK3xsWN62Gs%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_52.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_52.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=U0GubiVKqoTiVTMYOxOaK2zIvf0%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_53.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_53.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=EBt6cms9cwtUW3DVccyImXi2SlQ%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_54.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_54.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=PbhH1RnPq3sLi4NzhrbHOcWePow%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_55.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_55.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Aedqb5D%2FDg6aGIIiEj9rm0Awero%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_56.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_56.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=A89MXyRZws8osOtfjzZ7yQVon8U%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_57.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_57.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=yxFwS%2FBrBhuqcob0f9su8dqwQwc%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_58.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_58.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Zvn9SBpW3%2FF7iOyYdP6X1fJVG3Q%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_59.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_59.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=kPc0Tv6OzB3n3EPxy8hfZEXMz7I%3D&Expires=1657934024'

wget -O 20200705v1/full/pdf_parses/pdf_parses_6.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_6.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=b1OuRMRNbR4gY4XK818mVH%2FNUpU%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_60.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_60.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=X%2BvC1dRKYw%2BSCcecDHez5JHbMaE%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_61.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_61.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=orzn018z%2BvvNjbQamO205pkCUQw%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_62.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_62.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=a3I%2FN2Xfvr54aH7ydEPGRENuA%2BY%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_63.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_63.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=1b117N%2F%2BuhE92rPKwzQzKxamZxs%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_64.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_64.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=GrM0TOAcywxPhju%2BijNzPmlbNTI%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_65.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_65.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=cBtZijFTwJV3tvzmXSewqcqLrcA%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_66.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_66.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=EnmQ7Ey20QVSIIbA%2BHk6EAkHSdQ%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_67.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_67.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=14iVKisakgdl0jLGWEql68GvLx4%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_68.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_68.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=zVcpnFYW1%2Bwe%2BJgu6Z4w0eZGzbg%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_69.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_69.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=EV1qkRwS7gBmAlUQrdsv4afTk4Y%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_7.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_7.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=6SxYT7JELiV6Q5g6QWPtRFXFzU8%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_70.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_70.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=lEQO53q7aMFToh%2FBcytTHj2L7VU%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_71.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_71.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=0BuS%2BirdHuAeHMKy5YKGWuQ21Hs%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_72.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_72.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=PHGmR8sbFIVO8KJk53xlqcTLFPk%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_73.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_73.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=iZ1IpcGKURJR5RoKzCpfw39khkI%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_74.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_74.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=F7Q%2FIgRMot%2BC2CtnFVCuU8aKPXs%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_75.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_75.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=prfGzuL191eH9SKqdTpa7lK0aLE%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_76.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_76.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=15GZyOQovRWQe8ZtE7a7ws7ObP8%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_77.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_77.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=W4ZzgrjRWpX1N7OFSWZ4tLw%2BtNc%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_78.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_78.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=5kAb%2FyxSq4MXendHW7E%2F9vMqfRo%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_79.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_79.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=09Efn6pJa8md6BvCMAQeX%2BGY22c%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_8.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_8.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Q387rZr53KaZfTrRWK9T2G3NX%2Fo%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_80.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_80.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=mLyBPAd1OgEcKDdaABKc9dngcG0%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_81.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_81.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Y1dzbIqsXpNcuBFevF1Cm3s%2BQpE%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_82.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_82.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=YvEq3urfkZ%2BzqhU%2B4pMoP01GQLQ%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_83.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_83.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=o4BEFaF5iurte4EFOQ5cF0xMxL8%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_84.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_84.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=7kGreSKXukThLPBDeXzFbO5RnoU%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_85.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_85.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=ushO9uk7%2Fgehbj6Wt7EBlLYJxEQ%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_86.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_86.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=0YvUlgHNi609cxtx%2BjM1jzeF3HU%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_87.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_87.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=8%2Bc7tABcqtmS6XIFmZUFFLRw%2F3c%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_88.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_88.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=IMRi%2FGb01m3MDpqxhfIUA5FvLnM%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_89.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_89.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=2NV%2B7QiYVOXET9BF0cKi6JL3PCg%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_9.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_9.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=aYAxuANWk6ZbcT2MwmzDyyPbVNI%3D&Expires=1657934025'

wget -O 20200705v1/full/pdf_parses/pdf_parses_90.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_90.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Akcsc4KsyTcs18Lyg%2BiP3I%2BbD7Q%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_91.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_91.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=w8FCp2wd%2F%2FTyLZP2Z1IUZrwdh%2Fg%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_92.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_92.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=AxtNws2p2%2F81cTKqEvaWKmseFhI%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_93.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_93.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=vB3VI9xXqkFAAfHzgWKATPKt3Q8%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_94.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_94.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=3GYXNvQnyRYFUpPWn%2B9%2F2kvcTMo%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_95.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_95.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=oGpMBoQJ7IuQDz%2Fe2TPJ5er%2BvM4%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_96.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_96.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=Ozvtak%2FQMD3iKoPiuq7JI%2BrK8IE%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_97.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_97.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=37MOcGI7U26Nxg%2BW%2FizsyKWm%2BW8%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_98.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_98.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=BwlqTIn0UR2WJTBmbsqkIVXq4Jw%3D&Expires=1657934026'

wget -O 20200705v1/full/pdf_parses/pdf_parses_99.jsonl.gz 'https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_99.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=LctY7Jramv3wHSRG9wLRRPwX6k0%3D&Expires=1657934026'
