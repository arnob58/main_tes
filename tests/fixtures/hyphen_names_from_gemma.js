
const chat-history = document.querySelector('.chat-history');
const user-messages = document.querySelector('.user-messages');
const tessera-messages = document.querySelector('.tessera-messages');
const chat-text-input = document.querySelector('.chat-text-input');
const chat-send-button = document.querySelector('.chat-send-button');

chat-send-button.disabled = true;

async function send_message() {
    const message = chat-text-input.value.trim();
    if (!message) return;

    add_message('user', message);
    chat-text-input.value = '';

    show_thinking();
    chat-send-button.disabled = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data && data.reply) {
            add_message('tessera', data.reply);
        } else {
            add_message('tessera', 'Received an empty or invalid reply.');
        }

    } catch (error) {
        console.error('Error sending message:', error);
        add_message('tessera', `Error: Failed to get a response. (${error.message})`);
    } finally {
        remove_thinking();
        chat-send-button.disabled = false;
    }
}

chat-text-input.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        send_message();
    }
});

chat-send-button.addEventListener('click', send_message);

function add_message(sender, message) {
    const message_element = document.createElement('p');
    message_element.classList.add('chat-message');
    if (sender === 'user') {
        message_element.classList.add('user-message');
    } else if (sender === 'tessera') {
        message_element.classList.add('tessera-message');
    }
    message_element.textContent = message;
    chat-history.appendChild(message_element);
    chat-history.scrollTop = chat-history.scrollHeight;
}

function show_thinking() {
    const thinking_message = document.createElement('p');
    thinking_message.classList.add('chat-message', 'tessera-message');
    thinking_message.textContent = 'Tessera is thinking…';
    chat-history.appendChild(thinking_message);
    chat-history.scrollTop = chat-history.scrollHeight;
}

function remove_thinking() {
    const thinking_message = chat-history.lastElementChild;
    if (thinking_message && thinking_message.textContent === 'Tessera is thinking…') {
        chat-history.removeChild(thinking_message);
    }
}
