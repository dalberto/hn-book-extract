import asyncio
import argparse
import csv
from http import HTTPStatus
from typing import Dict, List
from io import StringIO

import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema import ChatResult
from langchain.text_splitter import CharacterTextSplitter


def scrape_hacker_news_comments(post_id: int) -> List[Dict]:
    url = f"https://news.ycombinator.com/item?id={post_id}"

    response = requests.get(url)

    if response.status_code != HTTPStatus.OK:
        raise Exception(f"Failed to retrieve webpage ({response.status_code}): {url}")

    soup = BeautifulSoup(response.content, "html.parser")

    comments = []
    for comment in soup.select(".athing.comtr"):
        comment_id = comment["id"]
        user = comment.select_one(".hnuser")
        age = comment.select_one(".age")
        text = comment.select_one(".commtext")

        if all([user, age, text]):
            comments.append(
                {
                    "id": comment_id,
                    "user": user.text,
                    "age": age.text,
                    "text": text.get_text(separator="\n", strip=True),
                }
            )

    return comments


def get_chat_prompt_template() -> ChatPromptTemplate:
    system_message_template = """You will be provided a list of user comments.
Extract all books as a CSV, omit quotes and use pipes as a separator.
Return only the CSV.

Input:
===
I really enjoyed reading The Lord of the Rings by J.R.R. Tolkien
I don't read much these days
Zero to One' by Peter Thiel, thank me later.
More of a podcast guy myself
You should check out Black Swan by Nassim Taleb.
===

Output:
title|author
The Lord of the Rings|J.R.R. Tolkien
Zero to One|Peter Thiel
Black Swan|Nassim Taleb

Input:
===
{comments}
===

Output:
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    return prompt


def parse_pipe_delimited_csv(csv_string) -> List[Dict]:
    csv_reader = csv.reader(StringIO(csv_string), delimiter="|")
    header = next(csv_reader)
    return [dict(zip(header, row)) for row in csv_reader]


def process_raw_output(raw_output: ChatResult) -> List[Dict]:
    items = []

    for result in raw_output.generations:
        raw_text = result[0].text
        items.extend(parse_pipe_delimited_csv(raw_text))

    return items


def get_number_of_tokens(model_name: str, _input: str) -> int:
    return len(tiktoken.encoding_for_model(model_name).encode(_input))


def write_csv(items: List[Dict], filename: str) -> None:
    if len(items) == 0:
        raise Exception("No items to write")

    with open(filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=items[0].keys())
        writer.writeheader()
        writer.writerows(items)


async def main(args: argparse.Namespace):
    prompt_template = get_chat_prompt_template()

    # This is not quite right, but close enough;
    # Langchain appends 'System: ' to the string, which is not part of the API request.
    system_prompt_tokens_used = get_number_of_tokens(
        model_name=args.model_name,
        _input=prompt_template.format_prompt(comments="").to_string(),  # format with no input, yielding only the prompt
    )
    remaining_tokens = args.model_token_limit - system_prompt_tokens_used
    input_token_budget = remaining_tokens // 2

    comments = scrape_hacker_news_comments(args.post_id)
    concatenated_comments = "\n".join([f'"{comment["text"].strip()}"' for comment in comments])

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=input_token_budget,
        chunk_overlap=0,
        encoding_name=tiktoken.encoding_for_model(args.model_name).name,
    )
    text_chunks = text_splitter.split_text(concatenated_comments)

    messages = [prompt_template.format_prompt(comments=chunk).to_messages() for chunk in text_chunks]
    chat = ChatOpenAI(temperature=0, model_name=args.model_name, request_timeout=args.timeout)
    raw_output = await chat.agenerate(messages)

    print("Total token usage: ", raw_output.llm_output["token_usage"]["total_tokens"])

    books = process_raw_output(raw_output)
    write_csv(books, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape and process Hacker News comments")
    parser.add_argument(
        "-m",
        "--model_name",
        default="gpt-3.5-turbo",
        help="Model name (default: gpt-3.5-turbo) see: https://platform.openai.com/docs/models",
    )
    parser.add_argument(
        "-l",
        "--model_token_limit",
        type=int,
        default=4096,
        help="Model token limit (default: 4096) see: https://platform.openai.com/docs/models",
    )
    parser.add_argument("-t", "--timeout", type=int, default=120, help="Request timeout (default: 120)")
    parser.add_argument("-p", "--post_id", type=int, required=True, help="Hacker News post ID")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Output file name (default: <post_id>.csv)",
    )

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"{args.post_id}.csv"

    asyncio.run(main(args))

    
