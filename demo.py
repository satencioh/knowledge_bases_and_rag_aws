import boto3
import streamlit as st


REGION_NAME = 'us-east-1'
# cliente de boto3
bedrock_client = boto3.client(
        'bedrock-agent-runtime', 
        region_name=REGION_NAME
    )

# configuracion del modelo y base de conocimiento
MODEL_ID = 'anthropic.claude-v2:1'
MODEL_ARN = f'arn:aws:bedrock:{REGION_NAME}::foundation-model/{MODEL_ID}'
KB_ID = 'ESPYQJ0ESV'

def get_answers(questions):
    # recuperamos la información como contexto y se la pasamos al LLM junto con la consulta del usuario para generar la respuesta requerida.
    response = bedrock_client.retrieve_and_generate(
        input={
            'text': questions # Preguntas que queremos hacer al asistente
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE', # se especifica la base de conocimientos
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': KB_ID, # ID de la base de conocimientos
                'modelArn': MODEL_ARN # ID del mode
            }
        },
    )
    return response

#Streamlit app
def main():
    st.set_page_config(page_title="RAG - Amazon Bedrock")
    st.subheader('Tu asistente virtual de confianza 🧡', divider='rainbow')

    #se mantiene el historial de chat de las preguntas y respuestas del usuario usando session_state de Streamlit
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['text'])

    questions = st.chat_input('Preguntame lo que quieras ....')

    if questions:
        with st.chat_message('user'):
            st.markdown(questions)
        st.session_state.chat_history.append({"role": 'user', "text": questions})

        response = get_answers(questions)
        answer = response['output']['text']

        with st.chat_message('assistant'):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": 'assistant', "text": answer})

        # Muestra el contexto y el documento fuente
        if response['citations'] and response['citations'][0]['retrievedReferences']:
            contexts = []
            doc_url = None
            for reference in response['citations'][0]['retrievedReferences']:
                contexts.append(reference['content']['text'])
                if not doc_url:
                    doc_url = reference['location']['s3Location']['uri']
            st.markdown(f"<span style='color:#FFDA33'>Context used: </span> {', '.join(contexts)}", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#FFDA33'>Source Document: </span> {doc_url}", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:red'>No Context", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
