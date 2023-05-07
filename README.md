# hn-book-extract

First, make sure you have [Docker installed](https://docs.docker.com/get-docker/).

Then:

1. Set `OPENAI_API_KEY` in your environment.

2. Run:

```shell
docker compose build
```

3. Then:

```shell
docker compose run app --post_id=<post_id>
```

