import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pd = dict()

    page_links = corpus[page]
    page_links_count = len(page_links)

    # extract all keys from corpus
    keys = list(corpus.keys())
    keys.sort()

    N = len(keys)

    if page_links_count:
        # page has links
        random_page_prob = (1 - damping_factor) / N
    else:
        # no links
        random_page_prob = 1 / N

    for page in keys:

        if page in page_links:
            follow_link_prob = damping_factor / page_links_count
        else:
            follow_link_prob = 0

        page_prob = random_page_prob + follow_link_prob
        pd[page] = round(page_prob, 4)
    
    return pd


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = dict()
    pages = list(corpus.keys())

    for page in pages:
        counts[page] = 0
    
    # pick random initial page
    page = random.choice(pages)
    counts[page] += 1

    for sample in range(n-1):

        # get probability distribution for the current page
        pd = transition_model(corpus, page, damping_factor)        
        
        # sample the next page from pd
        page = random.choices(list(pd.keys()), weights=pd.values(), k=1)[0]

        # update the page counts
        counts[page] += 1
    
    pageranks = counts.copy()

    for page in pages:
        # devide count by n to get the probability
        pageranks[page] /= n

    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageranks = dict()
    pages = list(corpus.keys())

    N = len(pages)

    # set initial pagerank (equal for all pages)
    for page in pages:
        pageranks[page] = 1 / N
    
    old_pageranks = pageranks.copy()
    
    EPSILON = 0.0001 # this matched the presented result values best
    
    while True:

        for page in pages:

            random_select_prob = (1.0 - damping_factor) / N

            follow_link_prob = 0

            for i in pages:

                page_links = corpus[i]

                if len(page_links):
                    if page in page_links:
                        follow_link_prob += (old_pageranks[i] / len(page_links))
                else:
                    # page i has no outgoing links
                    follow_link_prob += old_pageranks[i] / N
            
            follow_link_prob *= damping_factor

            new_pagerank = random_select_prob + follow_link_prob
            pageranks[page] = new_pagerank

            accurate_pages_count = 0

            for p in pages:
                if abs(pageranks[p] - old_pageranks[p]) < EPSILON:
                    accurate_pages_count += 1

            if accurate_pages_count == N:
                return pageranks

        old_pageranks = pageranks.copy()

if __name__ == "__main__":
    main()
