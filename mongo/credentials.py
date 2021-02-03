# OUT OF DATE, server doesn't exist anymore

account = "sacred"
pw = "sacredClient123"
ip = "34.77.184.43"
db_name = "model"

mongo_url = "mongodb://{}:{}@{}/{}?authMechanism=SCRAM-SHA-1".format(account, pw, ip, db_name)
