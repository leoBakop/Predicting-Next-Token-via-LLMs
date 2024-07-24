def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Ignore empty lines and comments
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Evaluate the value to convert it to the correct type
                try:
                    value = eval(value)
                except:
                    pass  # Keep value as string if eval fails

                config[key] = value
    return config
