from pathlib import Path

import scipy
import pandas as pd

def loadUserArtists(userArtistsFile: Path) -> scipy.sparse.csr_matrix:
    userArtists = pd.read_csv(userArtistsFile, sep="\t")
    userArtists.set_index(["userID", "artistID"], inplace=True)
    coo = scipy.sparse.coo_matrix(
        (
            userArtists.weight.astype(float),
            (
                userArtists.index.get_level_values(0),
                userArtists.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()


class ArtistRetriever:

    def __init__(self):
        self._artistsDataFrame = None

    def getArtistNameFromID(self, artist_id: int) -> str:
        return self._artistsDataFrame.loc[artist_id, "name"]

    def loadArtists(self, artistsFile: Path)-> None:
        artistsDataFrame = pd.read_csv(artistsFile, sep="\t")
        artistsDataFrame = artistsDataFrame.set_index("id")
        self._artistsDataFrame = artistsDataFrame







if __name__ == "__main__":
    # userArtistsMatrix = loadUserArtists(Path("../musicdata/user_artists.dat"))
    # print (userArtistsMatrix)

    artistRetriever = ArtistRetriever()
    artistRetriever.loadArtists(Path("../musicdata/artists.dat"))
    artist = artistRetriever.getArtistNameFromID(1)
    print(artist)