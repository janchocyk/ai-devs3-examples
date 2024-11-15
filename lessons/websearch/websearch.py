import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

from openAIservice import OpenAIService
import prompts


# Define the WebSearchService class
class WebSearchService:
    def __init__(self, allowed_domains: List[Dict[str, Any]]):
        self.allowed_domains = allowed_domains
        self.api_key = os.getenv('FIRECRAWL_API_KEY')  # Replace with your actual API key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    async def is_web_search_needed(self, user_message: str, openai_service) -> bool:
        '''
        Classification in RAG system, is web search is needed or not.
        '''
        print('Input (is_web_search_needed):', user_message)
        system_prompt = {
            "role": "system",
            "content": prompts.use_search_prompt  # This should be defined elsewhere
        }

        user_prompt = {
            "role": "user",
            "content": user_message
        }

        try:
            response = openai_service.completion(
                [system_prompt, user_prompt],
                model='gpt-4o'
            )

            if response.choices[0].message.content:
                print('Is web search needed?', response.choices[0].message.content)
                result = int(response.choices[0].message.content)
                print('Output (is_web_search_needed): ', result)
                if result == 1:
                    return True
                elif result == 0:
                    return False
                else:
                    raise ValueError('Unexpected response format')

            raise ValueError('Unexpected response format')
        except Exception as error:
            print('Error in WebSearchService:', error)
            return False

    async def generate_queries(
            self, 
            user_message: str, 
            openai_service
        ) -> Tuple[List[Dict[str, str]], str]:
        print('Input (generate_queries):', user_message)
        system_prompt = {
            "role": "system",
            "content": prompts.ask_domains_prompt(self.allowed_domains)  # This function should format the prompt
        }

        user_prompt = {
            "role": "user",
            "content": user_message
        }

        try:
            response = openai_service.completion(
                [system_prompt, user_prompt],
                model='gpt-4o-mini',
                json_mode=True
            )
            print(response)

            if response.choices[0].message.content:
                result = json.loads(response.choices[0].message.content)
                # Filter queries to only include allowed domains
                filtered_queries = [
                    query for query in result['queries']
                    if any(domain['url'] in query['url'] for domain in self.allowed_domains)
                ]
                print('Generated queries:', filtered_queries)
                thoughts = result.get('_thoughts', '')
                print('Output (generate_queries):', {'queries': filtered_queries, 'thoughts': thoughts})
                return filtered_queries, thoughts

            raise ValueError('Unexpected response format')
        except Exception as error:
            print('Error generating queries:', error)
            return [], ''

    async def search_web(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        print('Input (search_web):', queries)
        search_results = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for query in queries:
                q = query['q']
                url = query['url']
                task = asyncio.create_task(self._search_single_query(session, q, url))
                tasks.append(task)
            results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                search_results.append(result)

        print('Output (search_web):', search_results)
        return search_results

    async def _search_single_query(self, session, q: str, url: str) -> Dict[str, Any]:
        try:
            # Add site: prefix to the query using domain
            domain = url if url.startswith('http') else f'https://{url}'
            # domain = aiohttp.ClientSession()._parse_url(domain).host
            domain = urlparse(domain).netloc
            site_query = f"site:{domain} {q}"

            payload = {
                "query": site_query,
                "searchOptions": {
                    "limit": 6
                },
                "pageOptions": {
                    "fetchPageContent": False
                }
            }
            # async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.firecrawl.dev/v0/search',
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f'HTTP error! status: {response.status}')
                result = await response.json()

            print('siteQuery:', site_query)
            print('result:', result)

            if result.get('success') and result.get('data') and isinstance(result['data'], list):
                return {
                    'query': q,
                    'results': [
                        {
                            'url': item['url'],
                            'title': item['title'],
                            'description': item['description']
                        } for item in result['data']
                    ]
                }
            else:
                print(f'No results found for query: "{site_query}"')
                return {'query': q, 'results': []}
        except Exception as error:
            print(f'Error searching for "{q}":', error)
            return {'query': q, 'results': []}

    async def score_results(
            self, 
            search_results: List[Dict[str, Any]], 
            original_query: str, 
            openai_service
        ) -> List[Dict[str, Any]]:
        print('Input (score_results):', {'search_results': search_results, 'original_query': original_query})
        scored_results = []

        tasks = []
        for result in search_results:
            query = result['query']
            for item in result['results']:
                task = asyncio.create_task(self._score_single_result(item, query, original_query, openai_service))
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Remove None results
        results = [res for res in results if res]

        # Sort and filter the results
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        filtered_results = sorted_results[:3]

        print('Output (score_results):', filtered_results)
        return filtered_results

    async def _score_single_result(
            self, 
            item: Dict[str, Any], 
            query: str, 
            original_query: str, 
            openai_service
        ) -> Dict[str, Any]:
        user_message = f"""<context>
Resource: {item['url']}
Snippet: {item['description']}
</context>

The following is the original user query that we are scoring the resource against. It's super relevant.
<original_user_query_to_consider>
{original_query}
</original_user_query_to_consider>

The following is the generated query that may be helpful in scoring the resource.
<query>
{query}
</query>"""

        try:
            response = openai_service.completion(
                [
                    {"role": "system", "content": prompts.score_results_prompt},  # This should be defined elsewhere
                    {"role": "user", "content": user_message}
                ],
                model='gpt-4o-mini',
            )

            if response.choices[0].message.content:
                score_result = json.loads(response.choices[0].message.content)
                item['score'] = score_result.get('score', 0)
                print('Score for', item['url'], item['score'])
                print('Thoughts:', score_result.get('reason'))
                return item
            else:
                item['score'] = 0
                return item
        except Exception as error:
            print(f'Error scoring result {item["url"]}:', error)
            return None

    async def select_resources_to_load(
            self, 
            user_message: str, 
            filtered_results: List[Dict[str, Any]],
            openai_service
        ) -> List[str]:
        system_prompt = {
            "role": "system",
            "content": prompts.select_resources_to_load_prompt  # This should be defined elsewhere
        }

        filtered_resources = [
            {'url': r['url'], 'snippet': r['description']}
            for r in filtered_results
        ]
        user_prompt_content = f"""Original query: "{user_message}"
Filtered resources:
{json.dumps(filtered_resources, indent=2)}"""

        user_prompt = {
            "role": "user",
            "content": user_prompt_content
        }

        print('userPrompt:', user_prompt)

        try:
            response = openai_service.completion(
                [system_prompt, user_prompt],
                model='gpt-4o-mini',
                json_mode=True
            )

            if response.choices[0].message.content:
                print('Response content:', response.choices[0].message.content)
                result = json.loads(response.choices[0].message.content)
                selected_urls = result.get('urls', [])

                # Filter out URLs that aren't in the filtered results
                valid_urls = [
                    url for url in selected_urls
                    if any(r['url'] == url for r in filtered_results)
                ]

                return valid_urls

            raise ValueError('Unexpected response format')
        except Exception as error:
            print('Error selecting resources to load:', error)
            return []

    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        # Filter out URLs that are not scrappable based on allowed_domains
        scrappable_urls = []
        for url in urls:
            domain = url.split('//')[-1].split('/')[0].replace('www.', '')
            print('domain:', domain)
            allowed_domain = next((d for d in self.allowed_domains if d['url'] == domain), None)
            print('allowedDomain:', allowed_domain)
            if allowed_domain and allowed_domain.get('scrappable'):
                scrappable_urls.append(url)

        print('scrappableUrls:', scrappable_urls)

        scraped_results = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in scrappable_urls:
                task = asyncio.create_task(self._scrape_single_url(session, url))
                tasks.append(task)
            results = await asyncio.gather(*tasks)

        for result in results:
            if result and result['content']:
                scraped_results.append(result)

        return scraped_results

    async def _scrape_single_url(self, session, url: str) -> Dict[str, str]:
        try:
            payload = {
                "url": url,
                "formats": ["markdown"]
            }
            async with session.post(
                'https://api.firecrawl.dev/v0/scrape',
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f'HTTP error! status: {response.status}')
                scrape_result = await response.json()

            if scrape_result and scrape_result.get('markdown'):
                print('scrapeResult:', scrape_result)
                return {'url': url, 'content': scrape_result['markdown']}
            else:
                print(f'No markdown content found for URL: {url}')
                return {'url': url, 'content': ''}
        except Exception as error:
            print(f'Error scraping URL {url}:', error)
            return {'url': url, 'content': ''}

