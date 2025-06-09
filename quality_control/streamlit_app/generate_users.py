import yaml
import string
import random

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

data = {
    'credentials': {
        'usernames': {},
    },
    'cookie': {
        'expiry_days': 0,
        'key': generate_random_string(16),
        'name': generate_random_string(8)
    },
    'preauthorized': None
}

num_users = 25

for i in range(1, num_users+1):
    data['credentials']['usernames'][str(i)] = {
        'name': f'user{i}',
        'password': generate_random_string(8)
    }

with open('auth_config.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)
