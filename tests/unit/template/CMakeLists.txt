# halo_tokenizer is linked for the message-offset -> token-index test.
halo_add_test(test_template SOURCES test_template.cpp LIBS halo_template halo_tokenizer)
