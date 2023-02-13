from fastbloom_rs import BloomFilter, FilterBuilder

def test_bloom_sizes():
    # Test that the bloom filter sizes are as expected
    # The expected sizes are from the original bloombag implementation
    # https://hur.st/bloomfilter/?n=100000&p=0.01&m=&k=
    
    num_features = 100000
    error_rate = 0.01

    # Regular bloom filter
    bloom = BloomFilter(num_features, error_rate)
    size = len(bloom.get_bytes())*8
    assert size == 958506 + 22

    # Counting bloom filter
    builder = FilterBuilder(num_features, error_rate)
    builder.enable_repeat_insert(True)
    bloom = builder.build_counting_bloom_filter()  # type: CountingBloomFilter
    size = len(bloom.get_bytes())*8
    assert size == (958506 + 22)*4

if __name__ == "__main__":
    test_bloom_sizes()