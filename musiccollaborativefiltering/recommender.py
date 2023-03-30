from pathlib import Path
from typing import Tuple, List

import implicit
import scipy

from musiccollaborativefiltering.data import loadUserArtists, ArtistRetriever


class ImplicitRecommender:

    def __init__(self, artistRetriever: ArtistRetriever, implicitModel: implicit.recommender_base.RecommenderBase,):
        self.artistRetriever = artistRetriever
        self.implicitModel = implicitModel

    def fit(self, userArtistsMatrix: scipy.sparse.csr_matrix) -> None:
        self.implicitModel.fit(userArtistsMatrix)

    def recommend(self, userID: int, userArtistsMatrix: scipy.sparse.csr_matrix, n: int = 1,) -> Tuple[List[str], List[float]]:
        artistsIDs, scores = self.implicitModel.recommend(userID, userArtistsMatrix[n])
        artists = [
            self.artistRetriever.getArtistNameFromID(artistID)
            for artistID in artistsIDs
        ]
        return artists, scores


if __name__ == "__main__":

    #refers to data.py to pull user and artist data
    userArtists = loadUserArtists(Path("../musicdata/user_artists.dat"))

    #refers again to data.py to pull the artist data (name and ids)
    artistRetriever = ArtistRetriever()
    artistRetriever.loadArtists("../musicdata/artists.dat")

    #calls implicit als
    implicitModel = implicit.als.AlternatingLeastSquares(factors=50, iterations=10, regularization=0.01)

    #creates set of recommendations
    recommender = ImplicitRecommender(artistRetriever, implicitModel)
    recommender.fit(userArtists)
    artists, scores = recommender.recommend(5, userArtists, n=5)

    #prints recommendations
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")
