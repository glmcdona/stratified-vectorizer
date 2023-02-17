import time
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
    #assert size == 958506 + 22

    # Counting bloom filter
    builder = FilterBuilder(num_features, error_rate)
    builder.enable_repeat_insert(True)
    bloom = builder.build_counting_bloom_filter()  # type: CountingBloomFilter
    size = len(bloom.get_bytes())*8
    #assert size == (958506 + 22)*4

def test_bloom_bulk_insert_and_error():
    # Test that the bulk insert function works
    num_features = 100000
    error_rate = 0.01

    builder = FilterBuilder(num_features, error_rate)
    builder.enable_repeat_insert(True)

    for name, bloom in [
            ("BloomFilter", builder.build_bloom_filter()),
            ("CountingBloomFilter", builder.build_counting_bloom_filter())
        ]:
        print(f"\n--- Testing {name}")

        # Insert batch features
        start_time = time.time()
        features = [f"feature_{i}" for i in range(num_features)]
        bloom.add_str_batch(features)
        time_to_insert_batch = time.time() - start_time
        print(f"Time to insert {len(features)} features batched: {time_to_insert_batch}")

        # Check contains not batch
        start_time = time.time()
        for feature in features:
            assert bloom.contains(feature)
        print(f"Time to check {len(features)} features one at a time: {time.time() - start_time}")

        # Check contains batch
        start_time = time.time()
        batch_result = bloom.contains_str_batch(features)
        for result in batch_result:
            assert result
        print(f"Time to check {len(features)} features batched: {time.time() - start_time}")
        
        # Clear
        bloom.clear()

        # Insert features one at a time
        start_time = time.time()
        for feature in features:
            bloom.add_str(feature)
        time_to_insert_singleton = time.time() - start_time
        print(f"Time to insert {len(features)} features one at a time: {time_to_insert_singleton}")

        assert time_to_insert_batch < time_to_insert_singleton/10

        # Check that they are all present
        for feature in features:
            assert bloom.contains(feature)

        # Check that the false positive rate is as expected
        num_tests = 1000000
        expected_false_positives = error_rate * num_tests
        num_false_positives = 0
        for i in range(num_features, num_features+num_tests):
            feature = f"feature_{i}".encode()
            if bloom.contains(feature):
                num_false_positives += 1
        print(f"False positives: {num_false_positives} (expected: {expected_false_positives})")
        assert num_false_positives < expected_false_positives*1.1


if __name__ == "__main__":
    test_bloom_sizes()
    test_bloom_bulk_insert_and_error()