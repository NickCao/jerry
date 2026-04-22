FROM quay.io/ncao/jetpack-for-rhel:latest
COPY ./agent.container ./llm.container ./tts.container /usr/share/containers/systemd/
RUN ln -sr -t /usr/lib/bootc/bound-images.d/ /usr/share/containers/systemd/*.container
