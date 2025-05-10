from llm import query_llm, extract_tag, LLMResponse
from reddit import RedditToolset
from fire import Fire
import joblib as jl
import re
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from functools import lru_cache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
import httpx
import logging


logger = logging.getLogger(__name__)

# Rich console for better UX
console = Console()


class ResearchTool:
    """Main class for conducting Reddit research"""

    def __init__(self):
        self.reddit_cli = RedditToolset()
        self.cache_dir = os.path.join(os.getcwd(), ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(os.getcwd(), "results", self.session_id)
        os.makedirs(self.results_dir, exist_ok=True)
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pool = jl.Parallel(n_jobs=4, backend="threading", verbose=0)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ValueError, ConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: console.log(
            f"[yellow]Retrying LLM query (attempt {retry_state.attempt_number}/5)...[/yellow]"
        ),
    )
    def _query_llm_with_retry(self, prompt: str) -> LLMResponse:
        """Query LLM with retry logic"""
        try:
            response = query_llm(prompt)
            # Track tokens
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            return response
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise

    @lru_cache(maxsize=100)
    def _cached_query_llm(self, prompt: str) -> LLMResponse:
        """Cache LLM queries to avoid duplicates"""
        return self._query_llm_with_retry(prompt)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: console.log(
            f"[yellow]Retrying Reddit API call (attempt {retry_state.attempt_number}/5)...[/yellow]"
        ),
    )
    def _search_reddit_with_retry(
        self, query: str, subreddit: str = "", limit: int = 10
    ):
        """Search Reddit with retry logic

        Args:
            query: The search query string
            subreddit: Optional subreddit name to restrict search to (without r/ prefix)
            limit: Maximum number of results to return
        """
        try:
            return self.reddit_cli.search(query, subreddit=subreddit, limit=limit)
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            raise

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: console.log(
            f"[yellow]Retrying Reddit post fetch (attempt {retry_state.attempt_number}/5)...[/yellow]"
        ),
    )
    def _fetch_post_with_retry(self, post_id: str):
        """Fetch post with retry logic"""
        try:
            return self.reddit_cli.fetch_post_with_comments(post_id)
        except Exception as e:
            logger.error(f"Error fetching post {post_id}: {e}")
            raise

    def refine_prompt(self, initial_prompt: str) -> str:
        """Refine the initial user prompt with interactive questioning"""
        with console.status(
            "[bold green]Analyzing your research request...[/bold green]"
        ):
            analysis_template = f"""Given the following user prompt for Reddit research, analyze what additional information
            is needed to make the research more effective.

            <user_prompt>
            {initial_prompt}
            </user_prompt>

            Identify 2-3 specific questions that would help clarify the research goals.
            Format your response as a list of questions wrapped in <questions> tags.
            For example:
            <questions>
            1. What specific aspect of [topic] are you most interested in?
            2. Are you looking for personal experiences or expert opinions?
            3. What time period is most relevant for your research?
            </questions>
            """

            response = self._cached_query_llm(analysis_template)
            questions_text = extract_tag(response.content, "questions")

            if not questions_text:
                # If no questions were generated, proceed with the original prompt
                return initial_prompt

            questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

        # Present questions to the user
        console.print(
            Panel(
                Markdown("\n".join(questions)),
                title="[bold]To improve your research results, please answer these questions:[/bold]",
                border_style="yellow",
            )
        )

        # Collect answers
        answers = []
        for question in questions:
            # Remove numbering if present
            clean_question = re.sub(r"^\d+\.\s*", "", question)
            answer = Prompt.ask(f"[bold cyan]{clean_question}[/bold cyan]", default="")
            if answer:
                answers.append(f"Q: {clean_question}\nA: {answer}")

        # Combine original prompt with answers
        with console.status("[bold green]Refining your prompt...[/bold green]"):
            refine_template = f"""Given the following original user prompt and additional information,
            create a comprehensive, refined research prompt that clearly states the research goals.

            <original_prompt>
            {initial_prompt}
            </original_prompt>

            <additional_information>
            {"\n\n".join(answers)}
            </additional_information>

            Synthesize this information into a clear, detailed research prompt.
            """

            response = self._cached_query_llm(refine_template)
            refined_prompt = response.content

        console.print(
            Panel(
                Markdown(refined_prompt),
                title="[bold]Refined Research Prompt[/bold]",
                border_style="green",
            )
        )

        return refined_prompt

    def generate_search_plan(self, refined_prompt: str) -> List[str]:
        """Generate a search plan with queries based on refined prompt

        Supports both general queries and subreddit-specific queries in the format 'r/subreddit: query'
        """
        with console.status("[bold green]Generating search plan...[/bold green]"):
            final_plan_template = f"""Given the following research prompt, create a detailed plan for research on Reddit posts.
            <research_prompt>
            {refined_prompt}
            </research_prompt>

            The plan should contain a list of diverse search queries, optionally specifying a subreddit for targeted searches.
            Format your queries either as:
            1. Plain query: "search term" - This will search across all of Reddit
            2. Subreddit-specific query: "r/subreddit_name: search term" - This will search only in the specified subreddit

            Wrap all queries in <search_query> tags. Example:
                <search_query>
                search query 1
                r/science: search query 2
                r/askreddit: search query 3
                </search_query>

            Make sure to include a variety of queries that cover different aspects of the topic.
            Include both general searches and subreddit-specific searches where appropriate.
            Your queries should be ready to use - do not include placeholders.
            Queries should be short and simple, as Reddit's search is not very powerful.
            Create at least 10-20 different queries to ensure comprehensive coverage.
            """

            response = self._cached_query_llm(final_plan_template)
            final_plan = response.content
            search_queries = extract_tag(final_plan, "search_query").split("\n")
            # Filter out empty queries
            search_queries = [q.strip() for q in search_queries if q.strip()]

        console.print(
            Panel(
                "\n".join([f"• {q}" for q in search_queries]),
                title="[bold]Search Queries[/bold]",
                border_style="blue",
            )
        )

        return search_queries

    def search_posts(self, search_queries: List[str], limit: int = 10) -> List[Any]:
        """Search for posts using generated queries"""
        all_posts = []
        unique_post_ids = set()

        with Progress(
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[green]Searching Reddit...", total=len(search_queries)
            )

            def find_relevant_posts(query: str, limit: int = 10):
                try:
                    time.sleep(0.5)  # Rate limit to be nice to Reddit API

                    # Check if query is subreddit-specific (format: "r/subreddit: query")
                    subreddit = ""
                    search_query = query

                    if query.startswith("r/") and ":" in query:
                        parts = query.split(":", 1)
                        subreddit = parts[0].strip()
                        search_query = parts[1].strip()
                        # Strip the "r/" prefix as the API expects just the subreddit name
                        subreddit = subreddit[2:]

                    return self._search_reddit_with_retry(
                        search_query, subreddit=subreddit, limit=limit
                    )
                except Exception as e:
                    logger.error(f"Error searching for query '{query}': {e}")
                    return []

            jobs = [
                jl.delayed(find_relevant_posts)(query, limit)
                for query in search_queries
            ]

            # Create a callback to update progress
            class ProgressCallback:
                def __init__(self, progress_bar, task_id):
                    self.progress_bar = progress_bar
                    self.task_id = task_id
                    self.count = 0

                def __call__(self, arg):
                    self.count += 1
                    self.progress_bar.update(self.task_id, advance=1)
                    return arg

            callback = ProgressCallback(progress, task_id)
            posts = self.pool(callback(job) for job in jobs)

            # Flatten the list of posts and filter duplicates
            for post_list in posts:
                for post in post_list:
                    if post.id36 not in unique_post_ids:
                        all_posts.append(post)
                        unique_post_ids.add(post.id36)

        console.print(
            f"[bold green]Found {len(all_posts)} unique posts in total.[/bold green]"
        )

        # Save raw posts to cache
        post_data = [
            {"id": post.id36, "title": post.title, "url": post.permalink}
            for post in all_posts
        ]
        cache_path = os.path.join(self.cache_dir, f"posts_{self.session_id}.json")
        with open(cache_path, "w") as f:
            json.dump(post_data, f, indent=2)

        return all_posts

    def analyze_posts(
        self, all_posts: List[Any], refined_prompt: str
    ) -> List[Dict[str, str]]:
        """Fetch and analyze each post"""
        relevant_posts = []
        post_tokens = {"input": 0, "output": 0}  # Track tokens for post analysis

        with Progress(
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[green]Analyzing posts...", total=len(all_posts)
            )

            def fetch_and_analyze(post, task_id=None):
                try:
                    post_id = post.id36
                    permalink = f"{post.permalink}"
                    post_title = post.title

                    # Check cache first
                    cache_path = os.path.join(
                        self.cache_dir, f"analysis_{post_id}.json"
                    )
                    if os.path.exists(cache_path):
                        with open(cache_path, "r") as f:
                            data = json.load(f)
                            return {
                                "summary": data.get("summary", ""),
                                "input_tokens": data.get("input_tokens", 0),
                                "output_tokens": data.get("output_tokens", 0),
                                "permalink": data.get("permalink", permalink),
                                "title": data.get("title", post_title),
                            }

                    # Fetch and analyze if not in cache
                    post_content = self._fetch_post_with_retry(post_id)
                    template = f"""Given the following Reddit post, analyze if the content is relevant to the user research.
                    <reddit_post>
                    {post_content}
                    </reddit_post>

                    The research goals are:
                    <research_goals>
                    {refined_prompt}
                    </research_goals>

                    If the post is relevant to the research goals, write a summary of the content and comments and wrap in <summary> tags.
                    Otherwise, just write "no".
                    """
                    response = self._cached_query_llm(template)
                    summary = extract_tag(response.content, "summary")

                    # Cache the result with token info
                    result = {
                        "post_id": post_id,
                        "summary": summary,
                        "permalink": permalink,
                        "title": post_title,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    }

                    with open(cache_path, "w") as f:
                        json.dump(result, f)

                    return result
                except Exception as e:
                    logger.exception(
                        f"Error analyzing post {post.id36 if hasattr(post, 'id36') else 'unknown'}: {e}"
                    )
                    return {
                        "summary": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "permalink": "",
                        "title": "",
                    }

            # Use parallel processing with progress updates
            class ProgressCallback:
                def __init__(self, progress_bar, task_id):
                    self.progress_bar = progress_bar
                    self.task_id = task_id
                    self.count = 0

                def __call__(self, arg):
                    self.count += 1
                    self.progress_bar.update(self.task_id, advance=1)
                    return arg

            jobs = [jl.delayed(fetch_and_analyze)(post) for post in all_posts]
            callback = ProgressCallback(progress, task_id)
            results = self.pool(callback(job) for job in jobs)

            # Process results and track tokens
            post_results = []
            for result in results:
                summary = result.get("summary", "")
                if summary and summary.lower() != "no":
                    post_results.append(result)
                    post_tokens["input"] += result.get("input_tokens", 0)
                    post_tokens["output"] += result.get("output_tokens", 0)

            relevant_posts = post_results

        console.print(
            f"[bold green]Found {len(relevant_posts)}/{len(all_posts)} relevant posts.[/bold green]"
        )
        console.print(
            f"[bold blue]Post analysis used {post_tokens['input']} input tokens and {post_tokens['output']} output tokens.[/bold blue]"
        )

        # Save relevant posts to file
        results_path = os.path.join(self.results_dir, "relevant_posts.json")
        with open(results_path, "w") as f:
            json.dump(relevant_posts, f, indent=2)

        return relevant_posts

    def markdown_to_pdf(self, markdown_content: str, output_path: str) -> None:
        """Convert markdown content to PDF using pandoc.

        Args:
            markdown_content: The markdown text to convert
            output_path: The file path where the PDF will be saved
        """
        try:
            # Write markdown content to a temporary file
            import tempfile
            import os
            import subprocess
            import shutil

            # Check if pandoc is installed
            if shutil.which("pandoc") is None:
                logger.error("Pandoc is not installed. Cannot generate PDF.")
                console.print(
                    "[bold red]Error: Pandoc is not installed. PDF generation skipped.[/bold red]"
                )
                console.print(
                    "[bold yellow]To generate PDFs, please install Pandoc: https://pandoc.org/installing.html[/bold yellow]"
                )
                return False

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            ) as temp_md:
                temp_md.write(markdown_content)
                temp_md_path = temp_md.name

            try:
                cmd = [
                    "pandoc",
                    temp_md_path,
                    "-o",
                    output_path,
                    "--pdf-engine=pdflatex",
                    "--variable",
                    "colorlinks=true",
                    "--variable",
                    "linkcolor=blue",
                ]

                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Pandoc error: {e.stderr.decode() if e.stderr else str(e)}"
                )
                console.print(
                    "[bold red]Error running Pandoc. PDF generation failed.[/bold red]"
                )
                return False
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_md_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error converting markdown to PDF: {e}")
            return False

    def generate_report(
        self,
        relevant_posts: List[Dict[str, str]],
        refined_prompt: str,
        generate_pdf: bool = True,
    ) -> str:
        """Generate final research report

        Args:
            relevant_posts: List of post dictionaries with summary, permalink, and title
            refined_prompt: The refined research prompt
            generate_pdf: Whether to generate a PDF version of the report (default: True)
        """
        with console.status("[bold green]Generating final report...[/bold green]"):
            # Format post summaries with links for LLM input
            formatted_summaries = []
            for post in relevant_posts:
                summary = post.get("summary", "")
                title = post.get("title", "")
                permalink = post.get("permalink", "")
                formatted_summary = f"Post: {title}\nURL: {permalink}\n\n{summary}"
                formatted_summaries.append(formatted_summary)

            final_template = f"""Given the following summaries of reddit posts, write a research report.
            Research goals:
            <research_goals>
            {refined_prompt}
            </research_goals>
            <reddit_summaries>
            {"\n\n".join(formatted_summaries)}
            </reddit_summaries>
            The report should be concise and informative, highlighting the key findings and insights from the research.
            IMPORTANT: For each major finding or insight, include a link to the original Reddit post that discusses it using proper markdown link format [title](url).
            """
            response = self._cached_query_llm(final_template)
            final_report = response.content

        # Save final report as markdown
        report_path = os.path.join(self.results_dir, "final_report.md")
        with open(report_path, "w") as f:
            f.write(final_report)

        # Generate PDF if requested
        if generate_pdf:
            pdf_path = os.path.join(self.results_dir, "final_report.pdf")
            with console.status("[bold green]Converting report to PDF...[/bold green]"):
                if self.markdown_to_pdf(final_report, pdf_path):
                    console.print(f"[bold]PDF report saved to:[/bold] {pdf_path}")
                else:
                    console.print("[bold red]Failed to generate PDF report.[/bold red]")

        console.print(
            Panel(
                Markdown(final_report),
                title="[bold]Final Research Report[/bold]",
                border_style="green",
                expand=False,
            )
        )

        console.print(f"\n[bold]Report saved to:[/bold] {report_path}")
        console.print(
            f"[bold blue]Report generation used {response.input_tokens} input tokens and {response.output_tokens} output tokens.[/bold blue]"
        )

        return final_report


def main(initial_prompt: str, pdf: bool = True):
    """Main function to execute the research process

    Args:
        initial_prompt: The initial research prompt
        pdf: Whether to generate a PDF report (default: True)
    """
    console.rule("[bold blue]Reddit Research Tool[/bold blue]")

    if pdf:
        console.print("[bold green]PDF report generation is enabled.[/bold green]")

    try:
        research_tool = ResearchTool()

        # Step 1: Refine the prompt with interactive questioning
        refined_prompt = research_tool.refine_prompt(initial_prompt)

        # Step 2: Generate search plan based on refined prompt
        search_queries = research_tool.generate_search_plan(refined_prompt)

        # Step 3: Search for posts
        all_posts = research_tool.search_posts(search_queries)

        # Step 4: Analyze posts
        relevant_posts = research_tool.analyze_posts(all_posts, refined_prompt)

        # Step 5: Generate final report
        if relevant_posts:
            research_tool.generate_report(
                relevant_posts, refined_prompt, generate_pdf=pdf
            )
        else:
            console.print(
                "[bold red]No relevant posts found. Try refining your prompt and running again.[/bold red]"
            )

        # Report token usage statistics
        total_input = research_tool.total_input_tokens
        total_output = research_tool.total_output_tokens
        total_tokens = total_input + total_output

        console.rule("[bold blue]Token Usage Summary[/bold blue]")
        console.print(f"[bold green]Input tokens: {total_input:,}[/bold green]")
        console.print(f"[bold green]Output tokens: {total_output:,}[/bold green]")
        console.print(f"[bold green]Total tokens: {total_tokens:,}[/bold green]")

        # Save token usage to file
        token_path = os.path.join(research_tool.results_dir, "token_usage.json")
        with open(token_path, "w") as f:
            json.dump(
                {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total_tokens": total_tokens,
                },
                f,
                indent=2,
            )
        console.print(f"[bold]Token usage saved to:[/bold] {token_path}")

    except KeyboardInterrupt:
        console.print("\n[bold red]Research interrupted by user.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during research: {e}[/bold red]")


if __name__ == "__main__":
    Fire(main)
