docker run --name redis-server -p 6379:6379 -d redis redis-server --save 60 1 --loglevel warning
wget https://s3.amazonaws.com/redisinsight.download/public/latest/Redis-Insight-linux-x86_64.AppImage