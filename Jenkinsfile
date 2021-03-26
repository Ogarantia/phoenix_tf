pipeline {
    agent {
        label 'scaleway-gpu'
    }
    parameters {
        string(name: 'TF_VERSION', defaultValue: '2.4.0')
        string(name: 'QUIET', defaultValue: 'false', description: 'if true, no Slack message is sent on success')
    }
    triggers {
        // periodic runs with previous versions of TensorFlow
        parameterizedCron("@midnight %TF_VERSION=2.3.0;QUIET=true")
    }
    environment {
        SLACK_WEBHOOK = 'https://hooks.slack.com/services/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'
        REGISTRY_PROD = 'registryupstrideprod.azurecr.io'
        REGISTRY_DEV = 'registryupstridedev.azurecr.io'
        REPO = 'upstride'
        REPO_NAME = 'phoenix_tf'
        BUILD_TAG_PROD = "px"
        BUILD_TAG_DEV = "pxdev"
        TF_VERSION = "${params.TF_VERSION}"
        ARCH = "x86_64"
    }
    stages {
        stage ('git checkout'){
            steps{
                // checkout submodules and fetch toolchain binaries
                script {
                    sshagent(credentials : ['bitbucket-ssh2']) {
                        sh("""git checkout""")
                        sh("""git submodule update --init --recursive""")
                        sh("""cd core && git lfs fetch && git lfs checkout toolchain/gcc-8.4_8.4-1_${ARCH}.deb && cd ..""")
                    }
                }
                // set up environment: reading VERSION file after checkout
                script{
                    env.BUILD_VERSION = readFile("VERSION").trim()
                    env.BUILD_DEV = "${REGISTRY_DEV}/${REPO}:${BUILD_VERSION}-${BUILD_TAG_DEV}-tf${TF_VERSION}-gpu-${ARCH}"
                    env.BUILD_PROD = "${REGISTRY_PROD}/${REPO}:${BUILD_VERSION}-${BUILD_TAG_PROD}-tf${TF_VERSION}-gpu-${ARCH}"
                    env.GIT_AUTHOR_INFO = sh(script: "git show -s --pretty=%aN', '%aE", returnStdout: true).trim()
                    env.GIT_COMMIT_MSG = sh(script: "git show -s --pretty=%B", returnStdout: true).trim()
                }
            }
            post {
                always{
                    // prepare Slack message
                    header()
                }
                failure {
                    error("Cannot checkout")
                }
            }
        }
        stage('build docker') {
            steps {
                script {
                    sh("make distclean docker GPU=ON TF_VERSION=${TF_VERSION} DEVELOPMENT_DOCKER_REF=${BUILD_DEV} PRODUCTION_DOCKER_REF=${BUILD_PROD}")
                }
            }
            post {
                success {
                    success('Got docker images')
                }
                failure {
                    error("Cannot build docker images")
                }
            }
        }
        stage('run GPU unittests') {
            options {
                timeout(time: 300, unit: "SECONDS")
            }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}", 'registry-dev'){
                        docker.image(env.BUILD_DEV).inside(){
                            sh("PYTHONPATH=src/python python3 test.py")
                        }
                    }
                }
            }
            post {
                success {
                    success('GPU tests passed')
                }
                failure {
                    error("GPU unittests failed")
                }
            }
        }
        stage('run CPU unittests') {
            options {
                timeout(time: 300, unit: "SECONDS")
            }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}", 'registry-dev'){
                        docker.image(env.BUILD_DEV).inside(){
                            sh("CUDA_VISIBLE_DEVICES= PYTHONPATH=src/python python3 test.py")
                        }
                    }
                }
            }
            post {
                success {
                    success('CPU tests passed')
                }
                failure {
                    error("CPU unittests failed")
                }
            }
        }
        stage('promote development image') {
            when { branch 'master' }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        sh("""docker push $BUILD_DEV """)
                    }
                }
            }
            post {
                success {
                    success("Promoted developer image: $BUILD_DEV")
                }
                failure {
                    error("Promoting to development registry failed")
                }
            }
        }
        stage('promote image to staging') {
            when { branch 'master' }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_PROD}",'registry-prod'){
                        sh("""docker push $BUILD_PROD """)
                    }
                }
            }
            post {
                success {
                    success("Promoted image to staging: $BUILD_PROD")
                }
                failure {
                    error("Promoting to staging failed")
                }
            }
        }
    }
    post {
        success {
            script {
                info("All good. :heavy_check_mark:")
                info("Logs here: ${BUILD_URL}console")
                if (!params.QUIET)
                    slack()
            }
        }
        failure {
            script {
                info("Pipeline failed. :face_with_head_bandage:")
                info("Logs here: ${BUILD_URL}console")
                slack()
            }
        }
    }
}

def header(){
    env.SLACK_HEADER = ":glitch_crab: *`"+env.GIT_BRANCH+"`* updated by "+env.GIT_AUTHOR_INFO
    env.SLACK_MESSAGE = env.GIT_COMMIT_MSG+'\n\n'
}

def slack(){
    sh 'echo into slack :: - exiting -'
    DATA = '\'{"text":"'+env.SLACK_HEADER +'\n\n'+ env.SLACK_MESSAGE+'"}\''
    sh """
    curl -X POST -H 'Content-type: application/json' --data ${DATA} --url $SLACK_WEBHOOK
    """
}

def info(String body){
    env.SLACK_MESSAGE = env.SLACK_MESSAGE+'\n'+body.toString()
}

def success(String body){
    env.SLACK_MESSAGE = env.SLACK_MESSAGE+'\n :sunglasses: '+body.toString()
}

def error(String body){
    env.SLACK_MESSAGE = env.SLACK_MESSAGE+'\n :scream: *'+body.toString()+'*'
}
