a = {
    "key1": 1,
    "key2": 3,
    "key3": 2
}

# print(a.keys()) # ['key1', 'key2', 'key3']
# print(a.items()) # [('key1', 1), ('key2', 2), ('key3', 2)]
# print(a.values()) # [1, 2, 2]

sorted_values = [val for val, _ in sorted(a.items(), key=lambda item: item[1])]

print(sorted_values)