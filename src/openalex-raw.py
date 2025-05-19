from pathlib import Path

# tqdm is used to print a nice progress bar
# install it using `pip install tqdm`
from tqdm.auto import tqdm

import openalexraw as oaraw

# Path to the OpenAlex snapshot
openAlexPath = Path("<Location of the OpenAlex Snapshot>")

# Initializing the OpenAlex object with the OpenAlex snapshot path
oa = oaraw.OpenAlex(
    openAlexPath = openAlexPath
)

# Which entity to process
# "works" | "authors" | "institutions" | "venues" | "concepts"
entityType = "works"

# Getting the number of entries
entitiesCount = oa.getRawEntityCount(entityType)

# Iterating over all the entities of a certain type
for entity in tqdm(oa.rawEntities(entityType),total=entitiesCount):
    openAlexID = entity["id"]

    