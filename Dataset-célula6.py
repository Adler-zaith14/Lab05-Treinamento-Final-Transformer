dataset = load_dataset('bentrevett/multi30k', split='train')
subset  = dataset.select(range(1000))

print("Exemplo do dataset:")
print(subset[0])
