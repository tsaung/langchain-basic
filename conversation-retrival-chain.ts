import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

import {MessagesPlaceholder} from '@langchain/core/prompts'
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';

import dotenv from 'dotenv';
dotenv.config();

const createVectorStore = async () => {
  const loader = new CheerioWebBaseLoader(
    'https://js.langchain.com/docs/concepts/lcel'
  );
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new GoogleGenerativeAIEmbeddings();

  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  return vectorStore;
};

const createChain = async (vectorStore: MemoryVectorStore) => {
  const model = new ChatGoogleGenerativeAI({
    model: 'gemini-2.5-flash-lite',
    maxOutputTokens: 2048,
    temperature: 0.7,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      "Answer the user's question. Context: {context}. Question: {input}",
    ],
    new MessagesPlaceholder('chat_history'),
    ['user', '{input}'],
  ]);

  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  const retriever = vectorStore.asRetriever({
    k: 3,
  });

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder('chat_history'),
    ['user', '{input}'],
    [
      'user',
      'Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.'
    ]
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    retriever,
    llm: model,
    rephrasePrompt: retrieverPrompt,
  });
  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
  });

  return conversationChain;
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);
//Retrieve relevant documents

// Chat history
const chatHistory = [
  new HumanMessage('Hello'),
  new AIMessage('Hi there! How can I assist you today?'),
  new HumanMessage('My name is Thant Sin'),
  new AIMessage('Hi Leon, how can I help you'),
  new HumanMessage('What is LCEL?'),
  new AIMessage(
    'LCEL stands for LangChain Enhanced Language.'
  ),
];

const response = await chain.invoke({
  input: 'What is my name?',
  chat_history: chatHistory,
});
console.log(response);
