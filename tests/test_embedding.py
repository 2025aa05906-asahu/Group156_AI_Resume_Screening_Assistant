from services.embedding_service import EmbeddingModel


def test_embedding_generation():

    embedding = EmbeddingModel.generate_embedding("Python machine learning developer")

    assert embedding is not None
    assert len(embedding) > 0


def test_embedding_same_input_shape():

    embedding1 = EmbeddingModel.generate_embedding("Python developer")

    embedding2 = EmbeddingModel.generate_embedding("Machine learning developer")

    assert embedding1.shape == embedding2.shape
